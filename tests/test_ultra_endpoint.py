from nvidia.nemotron_ultra_agent import resolve_endpoint


def test_explicit_self_hosted_endpoint_wins(monkeypatch):
    monkeypatch.setenv('HF_TOKEN','unrelated-fixture')
    monkeypatch.setenv('NVIDIA_API_KEY','unrelated-nvidia-fixture')
    monkeypatch.setenv('CLAWD_INFERENCE_URL','http://127.0.0.1:8092/model/v1/')
    monkeypatch.setenv('CLAWD_MODEL','clawd-spark')
    monkeypatch.setenv('CLAWD_API_KEY','local-fixture')
    result=resolve_endpoint()
    assert result.base_url=='http://127.0.0.1:8092/model/v1'
    assert result.model=='clawd-spark'
    assert result.api_key=='local-fixture'


def test_portfolio_needs_observed_prices():
    from nvidia.nemotron_ultra_agent import run_portfolio_opt
    assert run_portfolio_opt(['SOL'])=={'error':'observed_price_history_required','available':False}


def test_paper_mode_rejects_nonpaper_command(monkeypatch):
    import nvidia.nemotron_ultra_agent as agent
    def forbidden(*args):raise AssertionError('Command must not run')
    monkeypatch.setattr(agent,'_vulcan',forbidden)
    assert 'rejected' in agent.execute_plan({'decision':'enter','vulcan_command':'vulcan wallet export'},'paper')
    assert 'no order' in agent.execute_plan({'decision':'hold','vulcan_command':'vulcan paper buy SOL'},'paper')
