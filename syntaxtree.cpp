#include "syntaxtree.h"

SyntaxTree::SyntaxTree(const string &line_tree)
{
	build_tree_from_str(line_tree);
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
		nodes.at(i).father = hidx;
		if (hidx == -1)
		{
			root_idx = i;
			nodes.at(i).father = i;			//TODO 检查是否必要
		}
		else
		{
			nodes.at(hidx).children.push_back(i);
		}
	}
}

