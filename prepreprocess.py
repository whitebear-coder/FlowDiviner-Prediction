from xml.dom import minidom
import networkx
import numpy as np
import dgl


def read_graph(filename='AbileneFlow'):
    # 初始化
    graph = networkx.Graph()
    betweeness_graph = networkx.Graph()
    node_index = 0
    nodes_dict = {}
    nodes_locate_list = []
    '''
    g = dgl.graph(([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 9, 9, 10, 12],
                   [1, 2, 3, 2, 3, 9, 10, 3, 4, 11, 13, 5, 11, 13, 7, 8, 9, 11, 10, 12, 14, 14, 10, 11, 12, 14]))
    g = dgl.add_self_loop(g)
    '''
    if filename == 'AbileneFlow':
        node_path = r'/home/hp/LZX/GMHCN/data/Abilene.xml'
    elif filename == 'BrainFlow':
        node_path = r'/home/hp/LZX/GMHCN/data/Brain.xml'
    elif filename == 'GeantFlow':
        node_path = r'/home/hp/LZX/GMHCN/data/geant.xml'
    doc = minidom.parse(node_path)
    # 取每个点的位置
    nodes = doc.getElementsByTagName("node")

    for node in nodes:
        sid = node.getAttribute("id")
        x = node.getElementsByTagName("x")[0]
        y = node.getElementsByTagName("y")[0]
        d = {node_index: sid}
        # node index 和 name 加入字典
        nodes_dict.update(d)
        # node locate --> nodes_locate_list
        nodes_locate_list.append((float(x.firstChild.data), float(y.firstChild.data)))
        # 节点加入图结构中
        # print(sid)
        graph.add_node(sid)
        # 更新
        node_index += 1

    source_list = np.array([])
    target_list = np.array([])
    # 求边的betweenness
    links = doc.getElementsByTagName("link")
    for link in links:
        source = link.getElementsByTagName("source")[0]
        target = link.getElementsByTagName("target")[0]
        capacity = link.getElementsByTagName("capacity")[0]

        for index, key in enumerate(nodes_dict):
            if source.firstChild.data == nodes_dict[key]:
                u = index
            if target.firstChild.data == nodes_dict[key]:
                v = index
        source_list = np.append(source_list, u)
        target_list = np.append(target_list, v)
        graph.add_edge(source.firstChild.data, target.firstChild.data, weight=capacity.firstChild.data)

    source_list = list(map(int, source_list.tolist()))
    target_list = list(map(int, target_list.tolist()))


    Abilene_graph = dgl.graph((source_list, target_list))
    Abilene_graph = dgl.add_self_loop(Abilene_graph)

    # 点节点度和边节点度
    node_betweenness = networkx.betweenness_centrality(graph)
    node_betweenness = list(node_betweenness.values())
    nodes_degree = dgl.DGLGraph.in_degrees(Abilene_graph)
    nodes_degree = np.array(nodes_degree / sum(nodes_degree))
    return Abilene_graph, nodes_degree, max(nodes_dict.keys()) + 1


if __name__ == '__main__':
    _, degree, nodes = read_graph('AbileneFlow')
    print(nodes)

