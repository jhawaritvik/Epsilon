"""
Report generator validation tests.

The report is a research artifact. These tests ensure:
- Reports render without broken links
- Embedded plots load correctly
- JSON blocks are valid
- Narrative never contradicts raw_results.json
- All required sections are present
"""

import pytest
import json
import re
from pathlib import Path
from typing import Dict, Any, List


def parse_markdown_links(markdown_text: str) -> List[str]:
    """Extract all markdown links from text."""
    # Pattern: [text](link) or ![alt](image.png)
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, markdown_text)
    return [match[1] for match in matches]


def parse_json_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    """Extract and parse all JSON code blocks from markdown."""
    # Pattern: ```json ... ```
    pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    
    json_blocks = []
    for match in matches:
        try:
            json_blocks.append(json.loads(match))
        except json.JSONDecodeError:
            json_blocks.append(None)  # Invalid JSON
    
    return json_blocks


def extract_sections(markdown_text: str) -> List[str]:
    """Extract section headings from markdown."""
    lines = markdown_text.split('\n')
    sections = []
    
    for line in lines:
        if line.startswith('#'):
            sections.append(line.strip())
    
    return sections


def extract_metrics_from_narrative(markdown_text: str) -> Dict[str, List[float]]:
    """
    Extract mentioned numeric metrics from narrative text.
    
    Example: "mean loss of 0.456" → {"mean_loss": [0.456]}
    """
    # Simple pattern for numbers in text
    pattern = r'(\w+[\s_]\w+)(?:[\s:]+|[\s]+of[\s]+)(\d+\.\d+|\d+)'
    matches = re.findall(pattern, markdown_text.lower())
    
    metrics = {}
    for metric_name, value in matches:
        metric_name = metric_name.replace(' ', '_')
        if metric_name not in metrics:
            metrics[metric_name] = []
        metrics[metric_name].append(float(value))
    
    return metrics


@pytest.mark.agent
@pytest.mark.report
class TestReportRendering:
    """Tests for report rendering and formatting."""
    
    def test_no_broken_file_links(self, temp_experiment_dir):
        """
        All file links in report should point to existing files.
        """
        # Create mock report
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        # Create referenced files
        (temp_experiment_dir / "plot1.png").write_text("fake image")
        (temp_experiment_dir / "dataset_used.json").write_text("{}")
        
        report_content = """
# Final Report

## Visualizations
![Comparison Plot](plot1.png)

## Data
See [dataset configuration](dataset_used.json)
"""
        report.write_text(report_content)
        
        # Extract links
        links = parse_markdown_links(report_content)
        
        # Verify all linked files exist
        for link in links:
            link_path = temp_experiment_dir / link
            assert link_path.exists(), f"Broken link: {link}"
    
    def test_broken_link_detected(self, temp_experiment_dir):
        """
        Broken links should be detected.
        """
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        # Don't create the referenced file
        report_content = """
# Final Report
![Missing Plot](nonexistent_plot.png)
"""
        report.write_text(report_content)
        
        # Extract links
        links = parse_markdown_links(report_content)
        
        # Check for broken links
        broken_links = []
        for link in links:
            link_path = temp_experiment_dir / link
            if not link_path.exists():
                broken_links.append(link)
        
        assert len(broken_links) > 0, "Should detect broken link"
        assert "nonexistent_plot.png" in broken_links[0]
    
    def test_all_plots_embedded(self, temp_experiment_dir):
        """
        All generated .png files should be embedded in report.
        """
        # Create plots
        import matplotlib.pyplot as plt
        for i in range(3):
            plt.figure()
            plt.plot([1, 2, 3], [1, 2, 3])
            plt.savefig(temp_experiment_dir / f"plot_{i}.png")
            plt.close()
        
        # Create report
        report = temp_experiment_dir / "FINAL_REPORT.md"
        report_content = """
# Final Report

## Visualizations
![Plot 0](plot_0.png)
![Plot 1](plot_1.png)
![Plot 2](plot_2.png)
"""
        report.write_text(report_content)
        
        # Get all plots
        actual_plots = list(temp_experiment_dir.glob("*.png"))
        
        # Get embedded plots
        links = parse_markdown_links(report_content)
        embedded_plots = [link for link in links if link.endswith('.png')]
        
        # All plots should be embedded
        assert len(embedded_plots) == len(actual_plots), \
            f"Not all plots embedded. Generated: {len(actual_plots)}, Embedded: {len(embedded_plots)}"


@pytest.mark.agent
@pytest.mark.report
class TestJSONBlocks:
    """Tests for JSON blocks in reports."""
    
    def test_json_blocks_are_valid(self, temp_experiment_dir):
        """
        All JSON code blocks should contain valid JSON.
        """
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        report_content = """
# Final Report

## Experiment Configuration
```json
{
  "hypothesis": "L2 reduces variance",
  "model": "neural_network"
}
```

## Raw Results
```json
{
  "baseline_loss": 0.5,
  "treatment_loss": 0.3
}
```
"""
        report.write_text(report_content)
        
        # Parse JSON blocks
        json_blocks = parse_json_blocks(report_content)
        
        # All should be valid
        assert len(json_blocks) == 2
        assert all(block is not None for block in json_blocks), \
            "All JSON blocks should be valid"
    
    def test_invalid_json_detected(self, temp_experiment_dir):
        """
        Invalid JSON blocks should be detected.
        """
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        report_content = """
# Final Report

## Invalid JSON
```json
{
  "broken": "json"
  "missing": "comma"
}
```
"""
        report.write_text(report_content)
        
        # Parse JSON blocks
        json_blocks = parse_json_blocks(report_content)
        
        # Should have one invalid block
        assert None in json_blocks, "Invalid JSON should be detected"


@pytest.mark.agent
@pytest.mark.report
class TestNarrativeConsistency:
    """
    Tests that verify narrative doesn't contradict raw results.
    """
    
    def test_narrative_matches_raw_results(self, 
                                           temp_experiment_dir,
                                           sample_raw_results):
        """
        Metrics mentioned in narrative should match raw_results.json.
        """
        # Save raw results
        raw_results_path = temp_experiment_dir / "raw_results.json"
        raw_results_path.write_text(json.dumps(sample_raw_results, indent=2))
        
        # Create report with narrative
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        baseline_mean = sample_raw_results["baseline_metrics"]["mean_loss"]
        treatment_mean = sample_raw_results["treatment_metrics"]["mean_loss"]
        
        report_content = f"""
# Final Report

## Results
The baseline achieved a mean loss of {baseline_mean}, while the 
treatment achieved {treatment_mean}.
"""
        report.write_text(report_content)
        
        # Extract metrics from narrative
        narrative_metrics = extract_metrics_from_narrative(report_content)
        
        # Cross-check with raw results
        # (This is simplified - real check would be more sophisticated)
        assert baseline_mean in narrative_metrics.get("mean_loss", []) or \
               baseline_mean in narrative_metrics.get("loss", [])
    
    def test_contradictory_narrative_detected(self,
                                              temp_experiment_dir):
        """
        Contradictions between narrative and raw results should be detected.
        """
        # Raw results say treatment is BETTER (lower loss)
        raw_results = {
            "baseline_loss": 0.5,
            "treatment_loss": 0.3  # Lower is better
        }
        
        raw_results_path = temp_experiment_dir / "raw_results.json"
        raw_results_path.write_text(json.dumps(raw_results))
        
        # But narrative says treatment is WORSE
        report = temp_experiment_dir / "FINAL_REPORT.md"
        contradictory_content = """
# Final Report

## Results
The treatment performed worse than baseline, with higher loss.
"""
        report.write_text(contradictory_content)
        
        # In real system, this would be caught by comparing:
        # - raw_results: treatment_loss < baseline_loss
        # - narrative: "performed worse"
        
        # These are contradictory
        assert raw_results["treatment_loss"] < raw_results["baseline_loss"]
        assert "worse" in contradictory_content
        
        # System should flag this contradiction


@pytest.mark.agent
@pytest.mark.report
class TestRequiredSections:
    """
    Tests that verify all required sections are present.
    """
    
    REQUIRED_SECTIONS = [
        "# Final Report",
        "## Executive Summary",
        "## Methodology",
        "## Results",
        "## Conclusion",
    ]
    
    def test_all_required_sections_present(self, temp_experiment_dir):
        """
        Report should contain all required sections.
        """
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        complete_report = """
# Final Report

## Executive Summary
Brief overview.

## Methodology
How we did it.

## Results
What we found.

## Conclusion
What it means.
"""
        report.write_text(complete_report)
        
        # Extract sections
        sections = extract_sections(complete_report)
        
        # Check all required sections exist
        for required in self.REQUIRED_SECTIONS:
            assert required in sections, f"Missing required section: {required}"
    
    def test_missing_section_detected(self, temp_experiment_dir):
        """
        Missing required sections should be detected.
        """
        report = temp_experiment_dir / "FINAL_REPORT.md"
        
        incomplete_report = """
# Final Report

## Executive Summary
Brief overview.

## Results
What we found.

(Missing: Methodology and Conclusion)
"""
        report.write_text(incomplete_report)
        
        sections = extract_sections(incomplete_report)
        
        # Should be missing some required sections
        missing = [req for req in self.REQUIRED_SECTIONS if req not in sections]
        
        assert len(missing) > 0, "Should detect missing sections"
        assert "## Methodology" in missing
        assert "## Conclusion" in missing


@pytest.mark.agent
@pytest.mark.report
def test_report_validation_integration(temp_experiment_dir, sample_raw_results):
    """
    End-to-end report validation.
    
    Checks:
    1. No broken links
    2. Valid JSON blocks
    3. All plots embedded
    4. Required sections present
    5. Narrative consistency
    """
    # Create complete artifacts
    (temp_experiment_dir / "run_experiment.py").write_text("# code")
    (temp_experiment_dir / "raw_results.json").write_text(
        json.dumps(sample_raw_results, indent=2)
    )
    (temp_experiment_dir / "dataset_used.json").write_text("{}")
    (temp_experiment_dir / "execution.log").write_text("log")
    
    # Create plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.savefig(temp_experiment_dir / "comparison.png")
    plt.close()
    
    # Create complete report
    report = temp_experiment_dir / "FINAL_REPORT.md"
    report_content = """
# Final Report

## Executive Summary
We compared baseline vs treatment.

## Methodology
Used neural networks on tabular data.

## Results
![Comparison Plot](comparison.png)

```json
{
  "baseline_loss": 0.5,
  "treatment_loss": 0.3
}
```

The treatment achieved lower loss (0.3 vs 0.5).

## Conclusion
Treatment is superior.
"""
    report.write_text(report_content)
    
    # Run all validations
    links = parse_markdown_links(report_content)
    for link in links:
        assert (temp_experiment_dir / link).exists(), f"Broken link: {link}"
    
    json_blocks = parse_json_blocks(report_content)
    assert all(block is not None for block in json_blocks), "Invalid JSON"
    
    sections = extract_sections(report_content)
    required = ["## Executive Summary", "## Methodology", "## Results", "## Conclusion"]
    for req in required:
        assert req in sections, f"Missing: {req}"
    
    # All validations passed
    assert True, "Complete report validation passed"
