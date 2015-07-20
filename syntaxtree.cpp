#include "syntaxtree.h"

SyntaxTree::SyntaxTree(const string &line_tree)
{
	build_tree_from_str(line_tree);
    cal_span_for_each_node(root_idx);
}

void SyntaxTree::build_tree_from_str(const string &line_tree)
{
	vector<string> wt_hidx_vec = Split(line_tree);
	sen_len = wt_hidx_vec.size();
	nodes.resize(sen_len);
	for (int i=0;i<sen_len;i++)
	{
		const string &wt_hidx = wt_hidx_vec.at(i);
		int sep = wt_hidx.rfind('_');
		string wt = wt_hidx.substr(0,sep);
		int hidx = stoi(wt_hidx.substr(sep+1));
		sep = wt.rfind('_');
		string word = wt.substr(0,sep);
		string tag = wt.substr(sep+1);
		nodes.at(i).word = word;
		nodes.at(i).tag = tag;
		nodes.at(i).idx = i;
		if (hidx == -1)
		{
			root_idx = i;
		}
		else
		{
			nodes.at(hidx).children.push_back(i);
		}
	}
}

/**************************************************************************************
 1. 函数功能: 计算当前子树覆盖的span
 2. 入口参数: 当前子树的根节点
 3. 出口参数: 无
 4. 算法简介: 无
************************************************************************************* */
void SyntaxTree::cal_span_for_each_node(int sub_root_idx)
{
	auto &node = nodes.at(sub_root_idx);
	if (node.children.empty() )                                           // 叶节点
	{
		node.src_span = make_pair(node.idx,0);
		return;
	}
	for (int child_idx : node.children)
	{
		cal_span_for_each_node(child_idx);
	}
	auto &first_child = nodes.at(node.children.front());
	auto &last_child = nodes.at(node.children.back());
    //首先合并第一个和最后一个孩子的源端span，然后与当前节点的源端span合并
	node.src_span = merge_span(merge_span(first_child.src_span,last_child.src_span),make_pair(node.idx,0));
}

/**************************************************************************************
 1. 函数功能: 将两个span合并为一个span
 2. 入口参数: 被合并的两个span
 3. 出口参数: 合并后的span
 4. 算法简介: 见注释
************************************************************************************* */
Span SyntaxTree::merge_span(Span span1,Span span2)
{
	if (span2.first == -1)
		return span1;
	if (span1.first == -1)
		return span2;
	Span span;
	span.first = min(span1.first,span2.first);												// 合并后span的左边界
	span.second = max(span1.first+span1.second,span2.first+span2.second) - span.first;		// 合并后span的长度，即右边界减去左边界
	return span;
}

