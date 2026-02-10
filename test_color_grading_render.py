#!/usr/bin/env python3
"""Test color grading rendering integration."""

import sys
import json
from pathlib import Path
from analyze_footage_creative import analyze_and_plan_creative_cut
from render_creative_cut import render_with_plan

def main():
    print("\n" + "="*70)
    print("🎬 COLOR GRADING INTEGRATION TEST")
    print("="*70)
    
    # Step 1: Analyze footage
    print("\n📊 Step 1: Analyzing footage and generating cut plan...")
    result = analyze_and_plan_creative_cut(target_duration=30)
    
    if "error" in result:
        print(f"❌ Analysis failed: {result['error']}")
        return False
    
    num_cuts = len(result.get("cuts", []))
    print(f"✅ Generated {num_cuts} cuts, {result['clips_analyzed']} clips analyzed")
    
    # Convert format for render_with_plan
    cut_plan = {
        "cut_plan": result.get("cuts", []),
        "target_duration": result.get("target_duration", 30),
        "actual_duration": sum(c.get("duration", 0) for c in result.get("cuts", [])),
        "num_cuts": num_cuts,
    }
    
    # Step 2: Render with different color grades
    print("\n🎨 Step 2: Testing color grading renders...")
    grades = ["none", "warm", "cool", "vibrant", "high_contrast", "cinematic"]
    
    results = {}
    for grade in grades:
        print(f"\n  Testing grade: {grade}...")
        render_result = render_with_plan(cut_plan, color_grade=grade)
        
        if render_result['success']:
            size_mb = render_result['file_size'] / (1024 * 1024)
            output_file = Path(render_result['output_file']).name
            results[grade] = {
                "status": "✅ SUCCESS",
                "file": output_file,
                "size_mb": size_mb,
                "cuts": render_result.get('total_cuts', 0),
            }
            print(f"    ✅ {grade}: {size_mb:.1f}MB ({output_file})")
        else:
            results[grade] = {
                "status": "❌ FAILED",
                "error": render_result.get("error", "Unknown error"),
            }
            print(f"    ❌ {grade}: {results[grade]['error']}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if "SUCCESS" in r["status"])
    failed_count = len(results) - success_count
    
    print(f"\n✅ Successful renders: {success_count}/{len(results)}")
    print(f"❌ Failed renders: {failed_count}/{len(results)}")
    
    for grade, res in results.items():
        if "SUCCESS" in res["status"]:
            print(f"  • {grade}: {res['size_mb']:.1f}MB ({res['cuts']} cuts)")
        else:
            print(f"  • {grade}: {res['error']}")
    
    # Save results
    results_file = Path("/home/codeai/montage-ai/data/output/color_grading_render_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📝 Results saved to: {results_file}")
    
    return failed_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
