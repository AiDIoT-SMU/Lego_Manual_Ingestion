"""
URL Handler: Downloads PDF instruction manuals from URLs.
Supports direct PDF links and handles common LEGO instruction URLs.
"""

import re
import tempfile
import requests
from pathlib import Path
from typing import Optional
from loguru import logger


class URLHandler:
    """Handles downloading PDF manuals from URLs."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize URL handler.

        Args:
            temp_dir: Directory for temporary downloads (default: system temp)
        """
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "lego_assembly_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"URL handler initialized with temp dir: {self.temp_dir}")

    def download_pdf(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Download PDF from URL to temporary directory.

        Args:
            url: URL to PDF file
            filename: Optional custom filename (default: extract from URL)

        Returns:
            Path to downloaded PDF file

        Raises:
            ValueError: If URL is invalid or not a PDF
            requests.RequestException: If download fails
        """
        logger.info(f"Downloading PDF from: {url}")

        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        if not filename:
            filename = self._extract_filename(url)

        if not filename.endswith('.pdf'):
            filename += '.pdf'

        output_path = self.temp_dir / filename

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                logger.warning(f"Content-Type is '{content_type}', may not be a PDF")

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Downloaded PDF to: {output_path} ({downloaded} bytes)")
            return output_path

        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            raise

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))

    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL."""
        path = url.split('?')[0]
        filename = path.split('/')[-1]

        if not filename or not filename.endswith('.pdf'):
            set_number = self._extract_lego_set_number(url)
            if set_number:
                filename = f"lego_instructions_{set_number}.pdf"
            else:
                url_hash = abs(hash(url)) % (10 ** 8)
                filename = f"manual_{url_hash}.pdf"

        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        return filename

    def _extract_lego_set_number(self, url: str) -> Optional[str]:
        """Extract LEGO set number from URL if present."""
        patterns = [
            r'(\d{7,8})\.pdf',
            r'/(\d{7,8})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def is_lego_url(self, url: str) -> bool:
        """Check if URL is from LEGO official site."""
        lego_domains = ['lego.com', 'lego.cn']
        return any(domain in url.lower() for domain in lego_domains)

    def cleanup(self):
        """Clean up temporary downloaded files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
