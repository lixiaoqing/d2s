#ifndef SYNTAXTREE_H
#define SYNTAXTREE_H
#include "stdafx.h"
#include "myutils.h"
#include "cand.h"
#include "vocab.h"

// 源端句法树节点
struct SyntaxNode
{
	string word;                                    // 该节点的词
	string tag;                                     // 该节点的词性
	int idx;										// 该节点在句子中的位置
	Span src_span;									// 以该节点为根节点的子树在句子中覆盖的span
	vector<int> children;							// 该节点的孩子节点在句子中的位置
	//CandOrganizer cand_organizer;                   // 组织该节点的翻译候选
	
	SyntaxNode ()
	{
		idx = -1;
	}
};

class SyntaxTree
{
	public:
		SyntaxTree(const string &line_of_tree);

	private:
		void build_tree_from_str(const string &line_of_tree);
		void cal_span_for_each_node(int sub_root_idx);

	public:
		int root_idx;
		int sen_len;
		vector<SyntaxNode> nodes;
};

#endif
