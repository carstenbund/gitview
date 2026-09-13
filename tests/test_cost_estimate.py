"""Tests for the pre-run cost estimate in `gitview analyze`."""

from gitview.commands.analyze import AnalyzeCommand


def _estimate(backend, model):
    cmd = AnalyzeCommand.__new__(AnalyzeCommand)   # the estimator uses no instance state
    return cmd._estimate_analysis_cost(commit_count=181, avg_msg_length=120,
                                       backend=backend, model=model)


def test_anthropic_current_sonnet_is_priced():
    est = _estimate('anthropic', 'claude-sonnet-5')
    expected = est['input_tokens'] / 1e6 * 2.00 + est['output_tokens'] / 1e6 * 10.00
    assert abs(est['cost_usd'] - expected) < 1e-9 and not est['plan_billed']


def test_claude_cli_is_plan_billed_with_api_equivalent():
    est = _estimate('claude-cli', 'sonnet')
    ref = _estimate('anthropic', 'claude-sonnet-5')
    assert est['plan_billed'] and est['cost_usd'] == 0.0
    assert abs(est['api_equivalent_usd'] - ref['cost_usd']) < 1e-9
    assert _estimate('claude-cli', 'haiku')['api_equivalent_usd'] < est['api_equivalent_usd']


def test_unknown_model_keeps_generic_fallback():
    est = _estimate('anthropic', 'claude-something-new')
    assert est['cost_usd'] > 0 and not est['plan_billed']
