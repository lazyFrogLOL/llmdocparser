def calculate_area(coords):
    """计算矩形面积"""
    return (coords[2] - coords[0]) * (coords[3] - coords[1])


def is_aligned_left(block1, block2, tolerance=10):
    """检查两个块的左上角x坐标是否对齐"""
    return abs(block1[1][0][0] - block2[1][0][0]) <= tolerance


def is_centered(block1, block2, tolerance=10):
    """检查两个块是否居中对齐"""
    center1 = (block1[1][0][0] + block1[1][0][2]) / 2
    center2 = (block2[1][0][0] + block2[1][0][2]) / 2
    return abs(center1 - center2) <= tolerance


def is_close_vertically(block1, block2, max_distance=100):
    """检查两个块在垂直方向上是否足够接近"""
    return block2[1][0][1] - block1[1][0][3] <= max_distance


def blocks_overlap(block1, block2):
    """检查两个块是否重叠"""
    x1, y1, x2, y2 = block1[1][0]
    x3, y3, x4, y4 = block2[1][0]
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def merge_blocks(block1, block2):
    """合并两个块"""
    new_coords = (
        min(block1[1][0][0], block2[1][0][0]),
        min(block1[1][0][1], block2[1][0][1]),
        max(block1[1][0][2], block2[1][0][2]),
        max(block1[1][0][3], block2[1][0][3])
    )
    new_order = min(block1[1][1], block2[1][1])
    if block1[0] == "text" and block2[0] == "text":
        new_key = "text"
    else:
        new_key = block1[0]+"_"+block2[0]
    return new_key, (new_coords, new_order)


def merge_consecutive_title_title_text(blocks):
    """合并连续的title、title、text块"""
    merged = []
    i = 0
    while i < len(blocks) - 2:
        if (blocks[i][0] == 'title' and
                blocks[i + 1][0] == 'title' and
                blocks[i + 2][0] == 'text' and
                blocks[i][1][1] + 1 == blocks[i + 1][1][1] and
                blocks[i + 1][1][1] + 1 == blocks[i + 2][1][1]):

            merged_block = merge_blocks(blocks[i], blocks[i + 1])
            merged_block = merge_blocks(merged_block, blocks[i + 2])
            merged.append(merged_block)
            i += 3
        else:
            merged.append(blocks[i])
            i += 1

    # 添加剩余的块
    merged.extend(blocks[i:])
    return merged


def is_in_corner(block, page_width, page_height, corner_threshold=100):
    """检查块是否在页面角落"""
    x1, y1, x2, y2 = block[1][0]
    return (x1 < corner_threshold and y1 < corner_threshold) or \
           (x2 > page_width - corner_threshold and y1 < corner_threshold) or \
           (x1 < corner_threshold and y2 > page_height - corner_threshold) or \
           (x2 > page_width - corner_threshold and y2 > page_height - corner_threshold)


def merge_title_and_text(blocks):
    """合并符合条件的title和text块"""
    merged = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == 'title' and i + 1 < len(blocks) and 'text' in blocks[i+1][0]:
            title_block = blocks[i]
            text_block = blocks[i+1]
            if (is_aligned_left(title_block, text_block) or is_centered(title_block, text_block)) and \
               is_close_vertically(title_block, text_block):
                merged.append(merge_blocks(title_block, text_block))
                i += 2
            else:
                merged.append(blocks[i])
                i += 1
        else:
            merged.append(blocks[i])
            i += 1
    return merged


def merge_figure_and_caption(blocks):
    """合并相邻的Figure和Figure caption块，或者Figure和Text块"""
    merged = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == 'figure' and i + 1 < len(blocks):
            if blocks[i+1][0] in ['figure_caption', 'figure', 'text']:
                merged.append(merge_blocks(blocks[i], blocks[i+1]))
                i += 2
            else:
                merged.append(blocks[i])
                i += 1
        else:
            merged.append(blocks[i])
            i += 1
    return merged



def merge_table_and_caption_or_reference(blocks):
    """合并相邻的Table和Table caption或Reference块"""
    merged = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == 'table' and i + 1 < len(blocks) and \
           blocks[i+1][0] in ['table caption', 'reference', 'text']:
            merged.append(merge_blocks(blocks[i], blocks[i+1]))
            i += 2
        else:
            merged.append(blocks[i])
            i += 1
    return merged


def filter_small_corner_blocks(blocks, page_width, page_height, min_area=200):
    """过滤掉面积小于min_area且位于页面角落的块"""
    return [block for block in blocks if not (calculate_area(block[1][0]) < min_area and
            is_in_corner(block, page_width, page_height))]


def merge_overlapping_blocks(blocks):
    """合并重叠的块"""
    merged = []
    i = 0
    while i < len(blocks):
        current_block = blocks[i]
        j = i + 1
        while j < len(blocks):
            if blocks_overlap(current_block, blocks[j]):
                current_block = merge_blocks(current_block, blocks[j])
                blocks.pop(j)
            else:
                j += 1
        merged.append(current_block)
        i += 1
    return merged


def merge_consecutive_texts(blocks):
    """合并连续的text块，不限制相邻"""
    merged = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == 'text':
            # 从当前 text 块开始，查找连续的 text 块
            j = i + 1
            while j < len(blocks) and blocks[j][0] == 'text':
                j += 1
            # 合并从 i 到 j 的所有 text 块
            merged_block = blocks[i]
            for k in range(i + 1, j):
                merged_block = merge_blocks(merged_block, blocks[k])
            merged.append(merged_block)
            i = j  # 更新 i 的值，跳过已合并的 text 块
        else:
            merged.append(blocks[i])
            i += 1
    return merged


def merge_equation_with_neighbors(blocks):
    """合并Equation类型的块与相邻的块"""
    merged = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == 'equation':
            to_merge = [blocks[i]]
            if i > 0:
                to_merge.append(blocks[i-1])
            if i < len(blocks) - 1:
                to_merge.append(blocks[i+1])
            merged_block = to_merge[0]
            for block in to_merge[1:]:
                merged_block = merge_blocks(merged_block, block)
            merged.append(merged_block)
            i += 1
        else:
            merged.append(blocks[i])
            i += 1
    return merged


def filter_small_header_footer(blocks, min_area=400):
    """过滤掉面积小于min_area的Header或Footer"""
    return [block for block in blocks if not (block[0] in ['header', 'footer'] and
            calculate_area(block[1][0]) < min_area)]


def is_block_contained(block1, block2):
    """检查block1是否包含于block2中"""
    x1, y1, x2, y2 = block1[1][0]
    x3, y3, x4, y4 = block2[1][0]
    return x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4


def remove_contained_blocks(blocks):
    """删除包含于其他块中的块"""
    filtered_blocks = []
    for i in range(len(blocks)):
        is_contained = False
        for j in range(len(blocks)):
            if i != j and is_block_contained(blocks[i], blocks[j]):
                is_contained = True
                break
        if not is_contained:
            filtered_blocks.append(blocks[i])
    return filtered_blocks


def merge_all(blocks, page_width, page_height):
    """应用所有合并规则和过滤规则"""
    if len(blocks) < 5:
        merged_block = blocks[0]
        for block in blocks[1:]:
            merged_block = merge_blocks(merged_block, block)
        return [merged_block]

    blocks = filter_small_corner_blocks(blocks, page_width, page_height)
    blocks = merge_overlapping_blocks(blocks)
    blocks = merge_title_and_text(blocks)
    blocks = merge_figure_and_caption(blocks)
    blocks = merge_table_and_caption_or_reference(blocks)
    blocks = merge_consecutive_texts(blocks)
    blocks = merge_equation_with_neighbors(blocks)
    blocks = filter_small_header_footer(blocks)
    blocks = merge_consecutive_title_title_text(blocks)
    blocks = remove_contained_blocks(blocks)
    blocks = merge_overlapping_blocks(blocks)
    return blocks
