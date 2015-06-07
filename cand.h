#ifndef CAND_H
#define CAND_H
#include "stdafx.h"
#include "ruletable.h"
#include "lm/left.hh"

//存储翻译候选
struct Cand	                
{
	//源端信息
	int rule_num;				//生成当前候选所使用的规则数目
	int glue_num;				//生成当前候选所使用的glue规则数目

	//目标端信息
	int tgt_word_num;			//当前候选目标端的单词数
	vector<int> tgt_wids;		//当前候选译文的单词id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;

	//来源信息, 记录候选是如何生成的
	RuleTrieNode* rule_node;                       // 生成当前候选的规则的源端
	vector<TgtRule>* matched_tgt_rules;            // 目标端非终结符个数及对齐相同的一组规则 TODO 为何要分组
	int rule_rank;                                 // 当前候选所用的规则在matched_tgt_rules中的排名
	vector<vector<Cand*> > cands_of_nt_leaves;     // 规则源端非终结符叶节点的翻译候选(glue规则所有叶节点均为非终结符)
	vector<int> cand_rank_vec;                     // 记录当前候选所用的每个非终结符叶节点的翻译候选的排名

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		rule_num = 1;
		glue_num = 0;

		tgt_word_num = 1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.clear();
		lm_prob = 0.0;

		rule_node = NULL;
		matched_tgt_rules = NULL;
		rule_rank = 0;
		cands_of_nt_leaves.clear();
		cand_rank_vec.clear();
	}
};

struct smaller
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//组织每个句法节点翻译候选的类
class CandOrganizer
{
	public:
		void add(Cand *&cand_ptr,int beam_size);
		Cand* top() { return data.front(); }
		Cand* at(size_t i) { return data.at(i);}
		int size() { return data.size();  }
		void sort() { std::sort(data.begin(),data.end(),larger); }
		void free();
	
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> data;                         // 当前节点所有的翻译候选
};

typedef priority_queue<Cand*, vector<Cand*>, smaller> Candpq;

#endif
