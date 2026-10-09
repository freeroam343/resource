# ============================================================
# INPUTS per frame
#   image            H x W x 3
#   cuboids          list of {id, class, corners_3d (8x3) in lidar frame}  (or 2D projected corners)
#   points           N x 3 lidar points in lidar frame
#   calib            T_lidar_to_cam (4x4), K (3x3), optional distortion
#   class_map        customer class -> set of compatible OneFormer thing classes
#
# OUTPUT per frame
#   panoptic         H x W int32, encoding (semantic_id, instance_id)
#   qa_report        per-instance scores + flags
# ============================================================

# ------------------------------------------------------------
# 0. ONE-TIME CALIBRATION SANITY CHECK (run on ~10 frames before anything else)
# ------------------------------------------------------------
for a few frames:
    uv, depth = project(points, calib)            # keep only depth > 0 and uv inside image
    overlay uv on image, colored by depth
    visually confirm points sit on the objects they should
    # if they don't, stop. Everything below assumes this is right.


# ------------------------------------------------------------
# 1. PRECOMPUTE GEOMETRY FOR THE FRAME
# ------------------------------------------------------------
uv_all, depth_all, valid = project(points, calib)
for each cuboid c:
    c.box2d   = tight_bbox( project(c.corners_3d, calib) )          # clip to image
    c.box2d   = expand(c.box2d, by=5%)                              # annotator boxes are often a bit tight
    inside3d  = points_inside_oriented_box(points, c.corners_3d)    # boolean N
    c.pts_uv  = uv_all[inside3d & valid]                            # projected lidar points on the object
    c.pts_d   = depth_all[inside3d & valid]
    c.depth   = median(c.pts_d) if len(c.pts_d) > 0 else depth_of_box_center(c)
    c.has_lidar = len(c.pts_uv) >= MIN_PTS   # e.g. 10


# ------------------------------------------------------------
# 2. RUN ONEFORMER ON THE FULL IMAGE
# ------------------------------------------------------------
seg_map, segments = oneformer_panoptic(image)
# seg_map: H x W segment id;  segments: {id -> {class, is_thing, score, area}}


# ------------------------------------------------------------
# 3. SELECT THE SEGMENT(S) THAT ARE "THE OBJECT INSIDE THE BOX"
# ------------------------------------------------------------
def candidate_segments(c):
    cands = []
    for s in segments where s.is_thing and s.class in class_map[c.class]:
        m          = (seg_map == s.id)
        in_box     = m & box_mask(c.box2d)
        containment = in_box.sum() / m.sum()          # fraction of segment inside box
        coverage    = in_box.sum() / area(c.box2d)    # fraction of box covered by segment
        lidar_hit   = fraction of c.pts_uv that land on m   if c.has_lidar else None

        if containment < 0.5:  continue     # mostly outside the box -> not this object (kills road/sky/buildings)
        if coverage    < 0.05: continue     # tiny sliver
        cands.append((s, containment, coverage, lidar_hit))
    return cands


def select_mask(c):
    cands = candidate_segments(c)
    if not cands: return None, "NO_SEGMENT"

    # primary ranking: lidar agreement if available, else coverage
    if c.has_lidar:
        cands.sort(by=lidar_hit, desc)
        best = cands[0]
        if best.lidar_hit < 0.3: return None, "LIDAR_DISAGREES"
    else:
        cands.sort(by=coverage*containment, desc)
        best = cands[0]

    mask = (seg_map == best.s.id)

    # absorb fragments: OneFormer sometimes splits one car into 2-3 segments
    for other in cands[1:]:
        if other.containment > 0.9 and other.lidar_hit > 0.5 and not overlaps_another_cuboid(other):
            mask |= (seg_map == other.s.id)

    # keep only connected components supported by lidar (or the largest if no lidar)
    comps = connected_components(mask)
    if c.has_lidar:
        mask = union of comps that contain >= 1 of c.pts_uv
    else:
        mask = largest(comps)

    # NEVER clip to the box. Only trim if leakage is extreme.
    outside = mask & ~box_mask(expand(c.box2d, by=15%))
    if outside.sum() > 0.3 * mask.sum():
        return None, "LEAKS_OUTSIDE_BOX"      # suspicious, send to rescue/review
    return mask, "OK"


# ------------------------------------------------------------
# 4. SPLIT MERGED INSTANCES
#    If the same OneFormer segment was selected by two cuboids, divide it.
# ------------------------------------------------------------
for each segment s claimed by cuboids C_s with |C_s| > 1:
    pix = pixels of s
    if all c in C_s have lidar:
        # assign each pixel to the cuboid whose projected lidar points are nearest (in image space)
        for each pixel p in pix:
            owner[p] = argmin_c  min_dist( p, c.pts_uv )
    else:
        # fallback: nearest box center, weighted by box size
        for each pixel p in pix:
            owner[p] = argmin_c  dist(p, center(c.box2d)) / diag(c.box2d)
    # optional: smooth owner map with a small majority filter to avoid speckle
    rewrite each c.mask = (owner == c)


# ------------------------------------------------------------
# 5. RESCUE PATH for cuboids with no acceptable mask
#    Point-prompted SAM2, NO box prompt.
# ------------------------------------------------------------
for each c with status != "OK":
    if not c.has_lidar: flag(c, "NEEDS_REVIEW"); continue

    pos = sample_spread(c.pts_uv, k=8)                 # spatially spread positive points on the object
    neg = lidar points within 1.5x box but NOT inside the 3D cuboid, k=4      # neighbours, ground
    mask, score = sam2(image, pos_points=pos, neg_points=neg)
    mask = largest_component_containing(mask, pos)
    if score < 0.7 or lidar_fraction(mask, c.pts_uv) < 0.5:
        flag(c, "NEEDS_REVIEW"); continue
    c.mask, c.status = mask, "RESCUED"


# ------------------------------------------------------------
# 6. RESOLVE OVERLAPS BETWEEN INSTANCES BY DEPTH (not by model score)
# ------------------------------------------------------------
instance_map = zeros(H, W)
depth_map    = +inf
for each c with a mask, sorted by c.depth DESCENDING:    # paint far first, near last overwrites
    instance_map[c.mask] = c.id
    depth_map[c.mask]    = c.depth
# near objects win contested pixels, matching real occlusion order


# ------------------------------------------------------------
# 7. STUFF FILL
# ------------------------------------------------------------
semantic_map = oneformer stuff classes from seg_map
semantic_map[instance_map > 0] = class of that instance        # things override stuff
# optional geometric cross-check for ground:
ground_pts  = ransac_ground_plane(points)
ground_uv   = project(ground_pts, calib)
if a stuff region labelled "building/vegetation" contains many ground_uv -> flag region


# ------------------------------------------------------------
# 8. QA SCORES  (compute for every instance, every frame)
# ------------------------------------------------------------
for each c:
    qa[c] = {
        box_iou        : iou(c.mask, box_mask(c.box2d)),            # low -> loose box OR bad mask
        lidar_in_mask  : fraction(c.pts_uv inside c.mask),          # low -> wrong object selected
        depth_spread   : std(depth under mask lidar points),        # high -> mask spans two objects
        status         : c.status,
        n_lidar        : len(c.pts_uv),
    }
    if lidar_in_mask < 0.6 or depth_spread > 2.0m or status != OK: flag(c)

# dataset-level: histogram these. A shift across ALL frames = calibration/annotation problem, not model.


# ------------------------------------------------------------
# 9. EMIT
# ------------------------------------------------------------
panoptic = encode_coco_panoptic(semantic_map, instance_map)
write png + json segment info (incl. iscrowd=1 for flagged/unresolvable regions)
write qa_report; route flagged frames to review tool


Two different things are going wrong, and the picture tells me which path each instance took.

The person is fine; the containers never went through OneFormer. The person silhouette is a clean model mask. The containers have horizontal/vertical scanline striping and holes, which is what you get when projected lidar points are being rasterized (dilated) directly into a mask. So almost certainly: candidate_segments() returned nothing for every container, and your rescue path is drawing the lidar points instead of using them as prompts. The far-right container is a perfect rectangle, which looks like the box fallback that we said should never exist, probably because it had too few lidar hits.

The root cause is the class gate. OneFormer’s vocabularies (COCO, ADE20K, Cityscapes) have no “shipping container.” It’ll label those as truck, building, wall, box, or, worst case, fold them into one big building/wall stuff segment together with the real building behind them. Any of those fails your s.is_thing and s.class in class_map["container"] test, so every container falls through.

What I’d change

Log status per instance and look at the histogram first. You’ll see all containers as NO_SEGMENT. Cheap confirmation before touching anything else.
Make the candidate test class-agnostic for classes OneFormer doesn’t know. For container, consider every segment (thing or stuff) and let containment, coverage and lidar agreement decide. If OneFormer gave each container its own truck/box segment, this alone fixes most of them. If it merged them into one wall segment with containment < 0.5, that segment is correctly rejected and you need step 3.
Fix the rescue path: lidar points are prompts, not pixels. Never rasterize them. Two good options, and for this scene I’d do both:
SAM 2 with lidar prompts, and negatives from the neighbours. Positives: 8–10 spatially spread points on the container face from inside its cuboid. Negatives: points from the adjacent containers’ cuboids and from the person’s cuboid. SAM 2 is very good at rigid boxy objects when told where the neighbours are; the earlier disappointment was the box prompt.
Depth gating for the leftover ambiguity. Interpolate a dense depth inside the 2D box from the lidar hits (nearest-neighbour or Delaunay is fine on a flat container face), then keep only pixels whose depth is within ±1 m of the container’s median depth. This removes the ground strip below the container, the sky above it, and anything nearer, including the person.
Use the 3D cuboid itself for the containers, if you have it. This is the big one. A container is a cuboid, so the projected visible faces of the 3D box (the polygon, not the tight 2D rectangle) are very close to the true outline already. Rasterize that polygon, subtract every nearer instance by depth, then snap the edges to image gradients with a few GrabCut iterations or a SAM 2 pass seeded from inside it. For people, vehicles and other non-boxy things the cuboid is just a selector, as before. If your annotations are only 2D rectangles, say so and I’ll adapt step 3 to compensate.
The person/container overlap is a painting-order problem. The person mask should win every contested pixel because it’s ~3 m away versus ~15 m. Make sure step 6 of the pseudocode (paint by depth, far first) runs after the rescue masks are produced, not before. Right now it looks like the containers were painted as full boxes and the person was then stamped on top, so where the person’s silhouette is slightly wrong you see container colour bleed. With depth-gated container masks the container simply has no pixels where the person stands, and the seam disappears.
Drop the box fallback entirely. For the far-right container with few lidar hits: use the 3D cuboid projection if available, else SAM 2 with whatever points you have, else flag. A rectangle is never an answer.

One more thing visible in the second image: the person’s box is enormous relative to the person. Your 5% box expansion is fine, but you should also check containment against the cuboid’s lidar footprint, not only the box, so an oversized box doesn’t let a background segment (the lamp post, the building edge) sneak in as a candidate.

If you share one frame with its cuboid JSON (I want to see whether it’s 3D corners or 2D rects), the point cloud and the calib, I’ll write the container branch properly: cuboid-face projection → depth gating → SAM 2 edge refinement → depth-ordered painting, and we can look at the result on this exact scene.
