from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from typing import List, Dict, Any

# 文本向量检索
def search_text_embedding(client: MilvusClient, query_text_emb, top_k=5, collection_name="fasion_multimodel_embedding", gender=None, product_type=None) -> List[Dict[str, Any]]:
    """
    在Milvus中根据文本向量进行检索
    Args:
        client: MilvusClient对象
        query_text_emb: 查询文本向量 (256,)
        top_k: 返回前k个结果
        gender: 性别过滤条件
        product_type: 产品类型过滤条件
        caption: 描述过滤条件
    Returns:
        检索结果列表，每个元素为dict
    """
    # 构建过滤条件
    filter_conditions = []
    if gender:
        filter_conditions.append(f'gender == "{gender}"')
    if product_type:
        filter_conditions.append(f'product_type == "{product_type}"')
    
    # 组合过滤条件
    filter_expr = None
    if filter_conditions:
        filter_expr = " and ".join(filter_conditions)
    
    search_params = {
        "collection_name": collection_name,
        "data": [query_text_emb.tolist()],
        "anns_field": "text_embedding",
        "search_params": {"metric_type": "IP"},
        "limit": top_k,
        "output_fields": ["pk", "caption", "product_type", "gender"]
    }
    
    # 只有在有过滤条件时才添加filter参数
    if filter_expr:
        search_params["filter"] = filter_expr
    
    results = client.search(**search_params)
    return results[0]  # 返回第一个查询的结果列表

# 图片向量检索
def search_image_embedding(client: MilvusClient, query_img_emb, top_k=5, collection_name="fasion_multimodel_embedding", gender=None, product_type=None) -> List[Dict[str, Any]]:
    """
    在Milvus中根据图片向量进行检索
    Args:
        client: MilvusClient对象
        query_img_emb: 查询图片向量 (256,)
        top_k: 返回前k个结果
        gender: 性别过滤条件
        product_type: 产品类型过滤条件
        caption: 描述过滤条件
    Returns:
        检索结果列表，每个元素为dict
    """
    # 构建过滤条件
    filter_conditions = []
    if gender:
        filter_conditions.append(f'gender == "{gender}"')
    if product_type:
        filter_conditions.append(f'product_type == "{product_type}"')
    
    # 组合过滤条件
    filter_expr = None
    if filter_conditions:
        filter_expr = " and ".join(filter_conditions)
    
    search_params = {
        "collection_name": collection_name,
        "data": [query_img_emb.tolist()],
        "anns_field": "image_embedding",
        "search_params": {"metric_type": "IP"},
        "limit": top_k,
        "output_fields": ["pk", "caption", "product_type", "gender"]
    }
    
    # 只有在有过滤条件时才添加filter参数
    if filter_expr:
        search_params["filter"] = filter_expr
    
    results = client.search(**search_params)
    return results[0]  # 返回第一个查询的结果列表

# 图文加权混合检索（简单加权融合）
def hybrid_search_weighted(
    client,
    query_img_emb,
    query_text_emb,
    alpha=0.5,
    top_k=5,
    collection_name="fasion_multimodel_embedding",
    gender=None,
    product_type=None
):
    """
    基于官方推荐的 WeightedRanker 实现文本和图片稠密向量加权混合检索
    """
    # 构建过滤条件
    filter_conditions = []
    if gender:
        filter_conditions.append(f'gender == "{gender}"')
    if product_type:
        filter_conditions.append(f'product_type == "{product_type}"')
    filter_expr = " and ".join(filter_conditions) if filter_conditions else None

    # 文本稠密向量检索
    dense_param = {
        "data": [query_text_emb.tolist()],
        "anns_field": "text_embedding",  # 修正：使用text_embedding而不是image_embedding
        "param": {"nprobe": 10},
        "limit": top_k * 2,
        "expr": filter_expr
    }
    request_dense = AnnSearchRequest(**dense_param)

    # 图片稠密向量检索
    reqs = [request_dense]
    if query_img_emb is not None:
        image_param = {
            "data": [query_img_emb.tolist()],
            "anns_field": "image_embedding",
            "param": {"nprobe": 10},
            "limit": top_k * 2,
            "expr": filter_expr
        }
        request_image = AnnSearchRequest(**image_param)
        reqs.append(request_image)

    # 配置加权融合策略
    rerank= WeightedRanker(alpha, 1-alpha)

    # 官方推荐的 hybrid_search 调用
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=rerank,
        limit=top_k,
        output_fields=["pk", "caption", "product_type", "gender"]
    )
    # 结果格式：results[0] 为最终融合后的 top_k
    return results[0]


