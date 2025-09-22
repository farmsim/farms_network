from farms_network.core.node import Node
from farms_network.models import Models
from farms_network.core.options import LINaPDannerNodeOptions
from farms_network.models.li_nap_danner_cy import LINaPDannerNodeCy


class LINaPDannerNode(Node):

    CY_NODE_CLASS = LINaPDannerNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.LI_NAP_DANNER, **kwargs)
