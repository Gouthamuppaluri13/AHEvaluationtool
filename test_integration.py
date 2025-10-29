#!/usr/bin/env python3
"""
Integration test for the Anthill AI+ Evaluation Tool
Tests that all components can be initialized and work together
"""

import os
import sys

# Set test environment variables
os.environ["TAVILY_API_KEY"] = "test_tavily_key"
os.environ["ALPHA_VANTAGE_KEY"] = "test_av_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from services.model_registry import ModelRegistry
        from models.online_predictor import OnlinePredictor
        from services.data_enrichment import DataEnrichmentService
        from services.fundraise_forecast import FundraiseForecastService
        from adapters.tavily_client import TavilyClient
        from adapters.indian_funding_adapter import IndianFundingAdapter
        from adapters.news_adapter import NewsAdapter
        from models.round_models import RoundLikelihoodModel, TimeToNextFinancingModel
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_online_predictor():
    """Test OnlinePredictor with heuristic fallback"""
    print("\nTesting OnlinePredictor...")
    
    try:
        from models.online_predictor import OnlinePredictor
        
        predictor = OnlinePredictor()
        
        test_inputs = {
            'age': 3,
            'total_funding_usd': 2000000,
            'team_size': 35,
            'num_investors': 6,
            'arr': 50000000,
            'burn': 8000000,
            'cash': 150000000
        }
        
        result = predictor.predict(test_inputs)
        
        assert 'round_probability_12m' in result
        assert 'round_probability_6m' in result
        assert 'meta' in result
        assert result['meta']['source'] == 'fallback'  # No model, should use fallback
        
        print(f"  12m probability: {result['round_probability_12m']:.2%}")
        print(f"  6m probability: {result['round_probability_6m']:.2%}")
        print("✓ OnlinePredictor working")
        return True
        
    except Exception as e:
        print(f"✗ OnlinePredictor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fundraise_forecast():
    """Test FundraiseForecastService"""
    print("\nTesting FundraiseForecastService...")
    
    try:
        from services.fundraise_forecast import FundraiseForecastService
        
        service = FundraiseForecastService()
        
        test_inputs = {
            'age': 3,
            'arr': 50000000,
            'burn': 8000000,
            'cash': 150000000,
            'team_size': 35,
            'num_investors': 6,
            'stage': 'Series A',
            'investor_quality_score': 7.5,
            'product_maturity_score': 8.0
        }
        
        forecast = service.predict(test_inputs)
        
        assert 'round_likelihood_6m' in forecast
        assert 'round_likelihood_12m' in forecast
        assert 'expected_time_to_next_round_months' in forecast
        
        print(f"  6m likelihood: {forecast['round_likelihood_6m']:.2%}")
        print(f"  12m likelihood: {forecast['round_likelihood_12m']:.2%}")
        print(f"  Expected time: {forecast['expected_time_to_next_round_months']:.1f} months")
        print("✓ FundraiseForecastService working")
        return True
        
    except Exception as e:
        print(f"✗ FundraiseForecastService failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_registry():
    """Test ModelRegistry"""
    print("\nTesting ModelRegistry...")
    
    try:
        from services.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        model_path = registry.load_model()
        
        # Should return None since no models are configured
        assert model_path is None
        
        print("  No models found (expected with no configuration)")
        print("✓ ModelRegistry working")
        return True
        
    except Exception as e:
        print(f"✗ ModelRegistry failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tavily_client():
    """Test TavilyClient initialization"""
    print("\nTesting TavilyClient...")
    
    try:
        from adapters.tavily_client import TavilyClient
        
        client = TavilyClient()
        
        # Should initialize but warn about missing key
        print("  Client initialized (API calls would require valid key)")
        print("✓ TavilyClient working")
        return True
        
    except Exception as e:
        print(f"✗ TavilyClient failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enrichment_service():
    """Test DataEnrichmentService initialization"""
    print("\nTesting DataEnrichmentService...")
    
    try:
        from services.data_enrichment import DataEnrichmentService
        
        service = DataEnrichmentService()
        
        # Check dataset summary
        summary = service.get_dataset_summary()
        print(f"  India index available: {summary.get('available', False)}")
        
        print("✓ DataEnrichmentService working")
        return True
        
    except Exception as e:
        print(f"✗ DataEnrichmentService failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Anthill AI+ Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_online_predictor,
        test_fundraise_forecast,
        test_model_registry,
        test_tavily_client,
        test_enrichment_service
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
