import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Dict, Tuple, Union, Set
from scipy.interpolate import splprep, splev
import sys

# =============================================================================
# PART 1: THE RECIPE MAKER (UNCHANGED)
# =============================================================================

class KnotBuilder:
    """Records a sequence of abstract knot-tying operations."""
    def __init__(self):
        self.recipe: List[Tuple] = []
        self._logical_segment_count = 1
        print("KnotBuilder Initialized: Ready to record recipe.")

    def cross(self, source: Union[str, int], crossing_type: str, target_segment_index: int):
        if not (1 <= target_segment_index <= self._logical_segment_count):
            raise ValueError(f"Invalid target index: {target_segment_index}.")

        if isinstance(source, int):
            if not (1 <= source <= self._logical_segment_count):
                raise ValueError(f"Invalid source index: {source}.")
            self.recipe.append(('cross', 'segment', source, crossing_type, target_segment_index))
            self._logical_segment_count += 2
            print(f"RECIPE: Adding 'Segment Cross' (Seg {source} {crossing_type} Seg {target_segment_index}).")

        elif isinstance(source, str) and source == 'end':
            self.recipe.append(('cross', 'end', crossing_type, target_segment_index))
            self._logical_segment_count += 2
            print(f"RECIPE: Adding 'End Cross' (Path extends {crossing_type} Seg {target_segment_index}).")
        
        else:
            raise TypeError("Invalid 'source'. Must be 'end' or a segment index (int).")

    def join(self):
        print("RECIPE: Adding 'Join'.")
        self.recipe.append(('join',))
        self._logical_segment_count += 1

    def analyze_recipe(self):
        print("\n--- Handing Recipe to Analyzer ---")
        analyzer = KnotAnalyzer(self.recipe)
        analyzer.find_redundant_operations()
        analyzer.estimate_crossing_number()

    def display(self, initial_length=3.0, loop_scale=2.0, verbose=False):
        print("\n--- Handing Recipe to Renderer ---")
        renderer = KnotRenderer(self.recipe, initial_length, loop_scale, verbose)
        renderer.render()

# =============================================================================
# PART 2: THE ANALYSIS ENGINE (UNCHANGED)
# =============================================================================

class KnotAnalyzer:
    def __init__(self, recipe: List[Tuple]):
        self.recipe = recipe
        self.LogicalSegment = lambda creator_op_index: {'creator': creator_op_index}
        self._redundant_indices = self._find_redundant_indices()

    def _find_redundant_indices(self) -> Set[int]:
        logical_path = [self.LogicalSegment(-1)]
        locked_op_indices = set()
        segment_cross_candidates = {}

        for op_index, operation in enumerate(self.recipe):
            if operation[0] != 'cross': continue
            
            target_idx = operation[-1]
            if 1 <= target_idx <= len(logical_path):
                creator_of_target = logical_path[target_idx - 1]['creator']
                if creator_of_target != -1:
                    locked_op_indices.add(creator_of_target)

            source_type = operation[1]
            if source_type == 'end':
                new_segments_for_target = [self.LogicalSegment(op_index), self.LogicalSegment(op_index)]
                logical_path[target_idx - 1 : target_idx] = new_segments_for_target
                logical_path.append(self.LogicalSegment(op_index))
            elif source_type == 'segment':
                source_idx = operation[2]
                segment_cross_candidates[op_index] = f"'{operation}'"
                indices = sorted([(source_idx, 'source'), (target_idx, 'target')], key=lambda x: x[0], reverse=True)
                for idx, _ in indices:
                    new_segments = [self.LogicalSegment(op_index), self.LogicalSegment(op_index)]
                    logical_path[idx - 1 : idx] = new_segments
        
        return set(segment_cross_candidates.keys()) - locked_op_indices

    def find_redundant_operations(self):
        print("\n--- Redundancy Analysis Report ---")
        segment_cross_ops = {i for i, op in enumerate(self.recipe) if op[0] == 'cross' and op[1] == 'segment'}
        if not segment_cross_ops:
            print("No 'segment cross' operations found to analyze for redundancy.")
            return

        for op_index in sorted(segment_cross_ops):
            description = f"Operation at step {op_index + 1}: {self.recipe[op_index]}"
            if op_index in self._redundant_indices:
                print(f"[REDUNDANT] {description}")
                print(f"  -> UNLOCKED: This loop is never used as a target by later steps.")
            else:
                print(f"[ESSENTIAL] {description}")
                print(f"  -> LOCKED IN: This loop is used as a target by a later step.")

    def estimate_crossing_number(self):
        print("\n--- Crossing Number Analysis ---")
        recipe_crossings = sum(1 for op in self.recipe if op[0] == 'cross')
        simplified_crossings = sum(1 for i, op in enumerate(self.recipe) if op[0] == 'cross' and i not in self._redundant_indices)
        
        print(f"Recipe Crossing Count (Upper Bound): {recipe_crossings}")
        print(f"Estimated Minimal Crossing Number: {simplified_crossings}")
        if recipe_crossings > simplified_crossings:
            print("  -> The simplification reduced the crossing count, giving a better estimate.")


# =============================================================================
# PART 3: THE BAKER (GEOMETRY & VISUALIZATION - BUG FIXED)
# =============================================================================

class KnotRenderer:
    class _Node:
        def __init__(self, node_id: int, position: np.ndarray): self.id = node_id; self.position = np.round(position, 2)
        def __repr__(self): return f"N{self.id}"
    class _Segment:
        def __init__(self, start_node, end_node): self.start_node = start_node; self.end_node = end_node
    
    def __init__(self, recipe, initial_length, loop_scale, verbose):
        self.recipe, self.verbose, self.loop_scale = recipe, verbose, loop_scale
        self._nodes, self._main_path_segments, self._node_counter = [], [], 0
        self.start_node = self._add_node([0, 0, 0]); self.end_node = self._add_node([initial_length, 0, 0])
        self._main_path_segments.append(self._Segment(self.start_node, self.end_node))
        if self.verbose: self._print_debug_state("Initial Geometric State")

    def _add_node(self, pos: List[float]):
        node = self._Node(self._node_counter, np.array(pos)); self._nodes.append(node); self._node_counter += 1; return node

    def render(self):
        for i, step in enumerate(self.recipe):
            op_type = step[0]
            if op_type == 'cross':
                _, source_type, *args = step
                if source_type == 'end':
                    crossing_type, target_id = args
                    self._execute_end_cross(crossing_type, target_id)
                elif source_type == 'segment':
                    source_id, crossing_type, target_id = args
                    self._execute_segment_cross(source_id, crossing_type, target_id)
            elif op_type == 'join':
                self._join()

            if self.verbose:
                self._print_debug_state(f"State After Interpreting Recipe Step {i+1}: {step}")
        
        self._plot_knot()

    def _execute_end_cross(self, crossing_type, target_idx):
        target_segment = self._main_path_segments[target_idx - 1]
        
        direction = target_segment.end_node.position - target_segment.start_node.position
        normal = np.array([-direction[1], direction[0], 0]); normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0,1,0])
        new_end_pos = target_segment.end_node.position + normal * self.loop_scale
        z_offset = 0.4 if crossing_type == 'over' else -0.4
        crossing_pos = (target_segment.start_node.position + target_segment.end_node.position) / 2.0; crossing_pos[2] += z_offset
        
        crossing_node = self._add_node(crossing_pos)
        new_end_node = self._add_node(new_end_pos)
        
        self._main_path_segments.append(self._Segment(self.end_node, crossing_node))
        self._split_segment(target_segment, crossing_node)
        self.end_node = new_end_node

    def _execute_segment_cross(self, source_idx, crossing_type, target_idx):
        source_segment, target_segment = self._main_path_segments[source_idx - 1], self._main_path_segments[target_idx - 1]
        
        midpoint_pos = (source_segment.start_node.position + source_segment.end_node.position) / 2.0
        knot_center = np.mean([n.position for n in self._nodes], axis=0)
        pull_vector = midpoint_pos - knot_center
        if np.linalg.norm(pull_vector) < 1e-6: pull_vector = np.array([0, 1, 0])
        pulled_node_pos = midpoint_pos + (pull_vector / np.linalg.norm(pull_vector)) * self.loop_scale
        z_offset = 0.4 if crossing_type == 'over' else -0.4
        crossing_pos = (target_segment.start_node.position + target_segment.end_node.position) / 2.0; crossing_pos[2] += z_offset
        
        pulled_node = self._add_node(pulled_node_pos)
        crossing_node = self._add_node(crossing_node)
        
        second_half_of_source = self._split_segment(source_segment, pulled_node)
        self._split_segment(target_segment, crossing_node)
        
        # Reroute the path: pulled_node -> crossing_node -> original_end_of_source
        self._main_path_segments.insert(self._main_path_segments.index(second_half_of_source), self._Segment(pulled_node, crossing_node))
        second_half_of_source.start_node = crossing_node
        
    def _split_segment(self, segment_to_split, new_node):
        original_start, original_end = segment_to_split.start_node, segment_to_split.end_node
        idx = self._main_path_segments.index(segment_to_split)
        segment_to_split.end_node = new_node # First half is modified in place
        second_half = self._Segment(new_node, original_end)
        self._main_path_segments.insert(idx + 1, second_half)
        return second_half

    def _join(self): 
        self._main_path_segments.append(self._Segment(self.end_node, self.start_node))
        self.end_node = None

    # ##### BUG FIX IS HERE #####
    def _get_ordered_path(self):
        """
        Traces the final path from the ordered list of segments. This is simpler
        and more robust than the previous adjacency list tracer.
        """
        if not self._main_path_segments:
            return []
        
        # The path is the start of the first segment, plus the end of every segment.
        path_coords = [self._main_path_segments[0].start_node.position]
        for seg in self._main_path_segments:
            path_coords.append(seg.end_node.position)
            
        # Defensive check: splprep needs at least k+1 (4) unique points.
        unique_path = [path_coords[0]]
        for point in path_coords[1:]:
            if not np.array_equal(point, unique_path[-1]):
                unique_path.append(point)
        
        if len(unique_path) < 4:
            print("Warning: Not enough unique points to generate a smooth plot.")
            return []

        return unique_path

    def _plot_knot(self):
        path_coords = self._get_ordered_path()
        if not path_coords: return # Exit if path tracing failed
        
        path_np = np.array(path_coords).T
        tck, u = splprep(path_np, s=0, per=True, k=3)
        u_fine = np.linspace(0, 1, 500)
        x, y, z = splev(u_fine, tck)
        
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=4, color='royalblue', label='Knot Path')
        ax.scatter(*self.start_node.position, c='green', s=150, label='Start/End', depthshade=True)
        
        node_counts = {}
        for seg in self._main_path_segments: node_counts[seg.start_node.id] = node_counts.get(seg.start_node.id, 0) + 1
        cross_nodes = [n for n in self._nodes if node_counts.get(n.id, 0) > 1]
        
        if cross_nodes: ax.scatter(*zip(*[n.position for n in cross_nodes]), c='red', s=80, label='Crossings', depthshade=True)
        ax.set_title("KnotGEN", fontsize=16); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); plt.show()
    
    def _print_debug_state(self, title: str):
        print("\n" + "="*50 + f"\nRENDERER DEBUG: {title.upper()}\n" + "="*50)
        end_node_str = repr(self.end_node) if self.end_node else 'Joined'
        print(f"Total Nodes: {len(self._nodes)}. Start: {self.start_node}, End: {end_node_str}")
        print("-" * 50 + "\nMain Path Segments (Geometric interpretation):")
        for i, seg in enumerate(self._main_path_segments): print(f"  Segment {i + 1}:  Connects {seg.start_node} -> {seg.end_node}")
        print("="*50 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    is_verbose = '--v' in sys.argv
    
    print("#" * 60)
    print("--- Analysis of a Recipe with a REDUNDANT Loop ---")
    print("#" * 60)
    unlocked_knot = KnotBuilder()
    unlocked_knot.cross(1, 'under', 1) 
    unlocked_knot.join()
    unlocked_knot.analyze_recipe()

    print("\n" + "#" * 60)
    print("--- Analysis of a Recipe with an ESSENTIAL Loop ---")
    print("#" * 60)
    locked_knot = KnotBuilder()
    locked_knot.cross(1, 'under', 1)
    locked_knot.cross('end', 'over', 2)
    locked_knot.join()
    locked_knot.analyze_recipe()

    print("\n" + "#" * 60)
    print("--- Building, Analyzing, and Rendering the Trefoil Knot ---")
    print("#" * 60)
    trefoil_builder = KnotBuilder()
    trefoil_builder.cross('end', 'under', target_segment_index=1)
    trefoil_builder.cross('end', 'over', target_segment_index=2)
    trefoil_builder.cross('end', 'under', target_segment_index=3)
    trefoil_builder.join()
    trefoil_builder.analyze_recipe()
    
    trefoil_builder.display(verbose=is_verbose)