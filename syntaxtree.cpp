#include "syntaxtree.h"

SyntaxTree::SyntaxTree(const string &line_of_tree)
{
	build_tree_from_str(line_of_tree);
	update_attrib(root);
}

void TreeStrPair::build_tree_from_str(const vector<string> &wt_hidx_vec)
{
	for (int i=0;i<src_sen_len;i++)
	{
		const string &wt_hidx = wt_hidx_vec.at(i);
		int sep = wt_hidx.rfind('_');
		string wt = wt_hidx.substr(0,sep);
		int hidx = stoi(wt_hidx.substr(sep+1));
		sep = wt.rfind('_');
		string word = wt.substr(0,sep);
		string tag = wt.substr(sep+1);
		src_nodes.at(i).word = word;
		src_nodes.at(i).tag = tag;
		src_nodes.at(i).idx = i;
		src_nodes.at(i).father = hidx;
		if (hidx == -1)
		{
			root_idx = i;
			src_nodes.at(i).father = i;			//TODO 检查是否必要
		}
		else
		{
			src_nodes.at(hidx).children.push_back(i);
		}
	}
}

