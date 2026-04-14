import os
import logging
import requests
import time

logger = logging.getLogger(__name__)


def is_edge_node():
    return os.environ.get('DAYU_NODE_ROLE') == 'edge'


def get_backend_address():
    return os.environ.get('DAYU_BACKEND_ADDRESS')


def edge_proxy_request(path, timeout=10):
    for _ in range(10):
        response = edge_proxy_request_url(path, timeout)
        if response:
            return response
        time.sleep(2)
    return None


def edge_proxy_request_url(path, timeout=10):
    backend_address = get_backend_address()
    if not backend_address:
        raise RuntimeError('DAYU_BACKEND_ADDRESS is not set for edge node')
    url = f'http://{backend_address}{path}'
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        logger.warning(f'Edge proxy request failed with status {response.status_code}: {url}')
    except Exception as e:
        logger.warning(f'Edge proxy request error: {e}')
    return None
