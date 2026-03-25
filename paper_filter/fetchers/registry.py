from __future__ import annotations

from paper_filter.fetchers.base import BaseFetcher


def get_fetcher(conference: str, year: int | None = None) -> BaseFetcher:
    """Return the appropriate fetcher for a conference and year.

    Some conferences use different sources depending on the year:
    - ICML: PMLR for 2021-2024, virtual site JSON for 2025+
    - NeurIPS: proceedings site for 2021-2024, virtual site JSON for 2025+
    """
    key = conference.lower().strip()

    if key in ("neurips", "nips"):
        if year is not None and year >= 2025:
            from paper_filter.fetchers.openreview_fetcher import VirtualSiteFetcher
            return VirtualSiteFetcher()
        from paper_filter.fetchers.neurips_fetcher import NeurIPSProceedingsFetcher
        return NeurIPSProceedingsFetcher()

    elif key == "iclr":
        from paper_filter.fetchers.openreview_api_fetcher import OpenReviewAPIFetcher
        return OpenReviewAPIFetcher()

    elif key == "icml":
        if year is not None and year >= 2025:
            from paper_filter.fetchers.openreview_fetcher import VirtualSiteFetcher
            return VirtualSiteFetcher()
        from paper_filter.fetchers.pmlr_fetcher import PMLRFetcher
        return PMLRFetcher()

    elif key in ("cvpr", "iccv"):
        from paper_filter.fetchers.cvf_fetcher import CVFFetcher
        return CVFFetcher()

    elif key == "eccv":
        from paper_filter.fetchers.ecva_fetcher import ECVAFetcher
        return ECVAFetcher()

    elif key == "aaai":
        from paper_filter.fetchers.aaai_fetcher import AAAIFetcher
        return AAAIFetcher()

    elif key == "emnlp":
        from paper_filter.fetchers.acl_fetcher import ACLFetcher
        return ACLFetcher()

    else:
        supported = ["neurips", "iclr", "icml", "cvpr", "iccv", "eccv", "aaai", "emnlp"]
        raise ValueError(
            f"Unknown conference: {conference}. Supported: {supported}"
        )
