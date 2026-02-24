#!/usr/bin/env python3
"""
Compare different visualization approaches for Step 1.
Run each visualization strategy to find the best one.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Digital Twin Visualization Comparison Tool")
print("=" * 70)
print("\nThis tool lets you test different visualization strategies for Step 1.")
print("Each approach enhances the rendering in different ways.")
print()

print("Available Visualization Strategies:")
print()
print("1. Enhanced Visualization")
print("   - Better color contrast (slightly adjusted colors)")
print("   - Black edge outlines for all bricks")
print("   - Light gray background")
print("   - Best for: Clear distinction between bricks")
print()
print("2. Unique Coloring")
print("   - Each brick gets unique brightness variation")
print("   - Colored sphere markers at brick centers")
print("   - Dark background")
print("   - Best for: Identifying individual bricks")
print()
print("3. Realistic Rendering")
print("   - Accurate LEGO colors")
print("   - Subtle ambient occlusion shading")
print("   - Neutral gray background")
print("   - Best for: True-to-life appearance")
print()
print("4. Original (Current Implementation)")
print("   - Uses current view_digital_twin.py")
print()
print("5. Test All (Sequential)")
print("   - Run all visualizations one after another")
print()

choice = input("Select visualization strategy (1-5) or 'q' to quit: ").strip()

if choice.lower() == 'q':
    print("Exiting...")
    sys.exit(0)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if choice == "1":
    print("\n→ Running Enhanced Visualization...")
    import test_enhanced_visualization
    test_enhanced_visualization.load_and_visualize_step1_enhanced()

elif choice == "2":
    print("\n→ Running Unique Coloring...")
    import test_colored_bricks_visualization
    test_colored_bricks_visualization.load_and_visualize_with_unique_colors()

elif choice == "3":
    print("\n→ Running Realistic Rendering...")
    import test_realistic_visualization
    test_realistic_visualization.load_and_visualize_realistic()

elif choice == "4":
    print("\n→ Running Original Visualization...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from view_digital_twin import visualize_digital_twin
    visualize_digital_twin(1)

elif choice == "5":
    print("\n→ Running All Visualizations Sequentially...")
    print("\nClose each window to proceed to the next visualization.\n")

    input("Press Enter to start with Enhanced Visualization...")
    import test_enhanced_visualization
    test_enhanced_visualization.load_and_visualize_step1_enhanced()

    input("\nPress Enter to continue to Unique Coloring...")
    import test_colored_bricks_visualization
    test_colored_bricks_visualization.load_and_visualize_with_unique_colors()

    input("\nPress Enter to continue to Realistic Rendering...")
    import test_realistic_visualization
    test_realistic_visualization.load_and_visualize_realistic()

    input("\nPress Enter to continue to Original...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from view_digital_twin import visualize_digital_twin
    visualize_digital_twin(1)

    print("\n✓ All visualizations complete!")

else:
    print(f"Invalid choice: {choice}")
    sys.exit(1)

print("\n" + "=" * 70)
print("Visualization Complete")
print("=" * 70)
print("\nFeedback:")
print("Which visualization worked best for distinguishing grey/white bricks?")
print("  1 = Enhanced")
print("  2 = Unique Coloring")
print("  3 = Realistic")
print("  4 = Original")
print()
