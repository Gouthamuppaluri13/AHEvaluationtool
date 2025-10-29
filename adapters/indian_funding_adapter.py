"""
Indian Funding News Adapter
Parses news from Indian startup ecosystem sources for funding trends.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from adapters.tavily_client import TavilyClient

logger = logging.getLogger(__name__)


# Indian startup news domains
INDIAN_STARTUP_DOMAINS = [
    "inc42.com",
    "yourstory.com",
    "entrackr.com",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "trak.in",
    "startupindia.gov.in"
]


class IndianFundingAdapter:
    """Extracts India-specific funding insights from news."""
    
    def __init__(self, tavily_client: Optional[TavilyClient] = None):
        self.tavily = tavily_client or TavilyClient()
    
    def get_sector_funding_trends(self, sector: str) -> Dict[str, Any]:
        """
        Get funding trends for a specific sector in India.
        
        Args:
            sector: Sector name (e.g., "FinTech", "HealthTech")
            
        Returns:
            Dictionary with:
            - round_counts_by_stage: Dict[stage, count]
            - median_round_size_inr: float
            - top_investors: List[str]
            - recent_rounds: List[Dict]
        """
        query = f"India {sector} startup funding rounds investment 2024 2025"
        
        results = self.tavily.search_news(
            query=query,
            include_domains=INDIAN_STARTUP_DOMAINS,
            max_results=15
        )
        
        if not results:
            logger.info(f"No news results found for sector: {sector}")
            return self._empty_trends()
        
        # Parse results
        trends = self._parse_funding_news(results)
        
        return trends
    
    def get_company_funding_news(self, company_name: str) -> List[Dict[str, Any]]:
        """
        Get recent funding news for a specific company.
        
        Args:
            company_name: Company name
            
        Returns:
            List of recent funding news items
        """
        query = f"{company_name} India funding raise investment"
        
        results = self.tavily.search_news(
            query=query,
            include_domains=INDIAN_STARTUP_DOMAINS,
            max_results=5
        )
        
        return results
    
    def _parse_funding_news(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse funding information from news results."""
        
        round_counts = {}
        amounts_inr = []
        investors = []
        recent_rounds = []
        
        for result in results:
            content = result.get('content', '') + ' ' + result.get('title', '')
            
            # Extract funding stages
            stages = self._extract_stages(content)
            for stage in stages:
                round_counts[stage] = round_counts.get(stage, 0) + 1
            
            # Extract amounts
            amount = self._extract_amount_inr(content)
            if amount:
                amounts_inr.append(amount)
            
            # Extract investors
            invs = self._extract_investors(content)
            investors.extend(invs)
            
            # Build recent round entry
            if stages or amount or invs:
                recent_rounds.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'stage': stages[0] if stages else 'Unknown',
                    'amount_inr': amount,
                    'investors': invs[:3]  # Top 3
                })
        
        # Calculate median
        median_amount = 0
        if amounts_inr:
            import statistics
            median_amount = statistics.median(amounts_inr)
        
        # Count top investors
        from collections import Counter
        top_investors = [inv for inv, count in Counter(investors).most_common(10)]
        
        return {
            'round_counts_by_stage': round_counts,
            'median_round_size_inr': median_amount,
            'top_investors': top_investors,
            'recent_rounds': recent_rounds[:5]  # Top 5 most recent
        }
    
    def _extract_stages(self, text: str) -> List[str]:
        """Extract funding stage mentions from text."""
        stages = []
        text_lower = text.lower()
        
        stage_patterns = [
            ('Pre-Seed', ['pre-seed', 'preseed', 'pre seed']),
            ('Seed', ['seed round', 'seed funding', 'seed stage']),
            ('Series A', ['series a', 'series-a']),
            ('Series B', ['series b', 'series-b']),
            ('Series C', ['series c', 'series-c']),
            ('Series D', ['series d', 'series-d']),
            ('Bridge', ['bridge round', 'bridge funding']),
            ('Growth', ['growth round', 'growth stage'])
        ]
        
        for stage_name, patterns in stage_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    stages.append(stage_name)
                    break
        
        return stages
    
    def _extract_amount_inr(self, text: str) -> Optional[float]:
        """Extract funding amount in INR from text."""
        
        # Patterns for amounts
        patterns = [
            r'₹\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakh|million|m|billion|bn)',
            r'INR\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakh|million|m|billion|bn)',
            r'(\d+(?:\.\d+)?)\s*(crore|cr|lakh)\s*rupees?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Convert to INR
                if 'crore' in unit or 'cr' == unit:
                    return value * 10000000  # 1 crore = 10M
                elif 'lakh' in unit:
                    return value * 100000  # 1 lakh = 100K
                elif 'million' in unit or unit == 'm':
                    return value * 1000000
                elif 'billion' in unit or 'bn' in unit:
                    return value * 1000000000
        
        # Try USD patterns and convert
        usd_patterns = [
            r'\$\s*(\d+(?:\.\d+)?)\s*(million|m|billion|bn)',
            r'USD\s*(\d+(?:\.\d+)?)\s*(million|m|billion|bn)',
        ]
        
        for pattern in usd_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Convert USD to INR (approximate rate: 1 USD = 83 INR)
                usd_value = value
                if 'million' in unit or unit == 'm':
                    usd_value *= 1000000
                elif 'billion' in unit or 'bn' in unit:
                    usd_value *= 1000000000
                
                return usd_value * 83  # Convert to INR
        
        return None
    
    def _extract_investors(self, text: str) -> List[str]:
        """Extract investor names from text."""
        investors = []
        
        # Common investor patterns
        patterns = [
            r'(?:led by|invested by|backed by|from)\s+([A-Z][A-Za-z\s&]+(?:Capital|Ventures|Partners|Fund|Investments?))',
            r'([A-Z][A-Za-z\s&]+(?:Capital|Ventures|Partners|Fund|Investments?))\s+(?:led|invested|backed)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                investor = match.strip()
                if len(investor) > 3 and len(investor) < 50:  # Sanity check
                    investors.append(investor)
        
        return list(set(investors))  # Remove duplicates
    
    def _empty_trends(self) -> Dict[str, Any]:
        """Return empty trends structure."""
        return {
            'round_counts_by_stage': {},
            'median_round_size_inr': 0,
            'top_investors': [],
            'recent_rounds': []
        }
