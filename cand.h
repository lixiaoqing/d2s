#ifndef CAND_H
#define CAND_H
#include "stdafx.h"
#include "ruletable.h"
#include "lm/left.hh"

//生成候选所使用的规则信息
struct Rule
{
	int nt_num;				  //规则中非终结符个数
	vector<int> src_ids;      //规则源端符号（包括终结符和非终结符）id序列
	TgtRule *tgt_rule;        //规则目标端
	int tgt_rule_rank;		  //该目标端在源端相同的所有目标端中的排名
	vector<int> tgt_nt_idx_to_src_sen_idx;  //目标端变量的位置在源端句子中对应的位置
};

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
	Rule applied_rule;          					//生成当前候选所使用的规则
	vector<vector<Cand*> > cands_of_nt_leaves;      // 规则源端非终结符叶节点的翻译候选(glue规则所有叶节点均为非终结符)
	vector<int> cand_rank_vec;                      // 记录当前候选所用的每个非终结符叶节点的翻译候选的排名

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

		cands_of_nt_leaves.clear();
		cand_rank_vec.clear();
	}
};

bool smaller( const Cand *pl, const Cand *pr );

struct cmp
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//组织每个句法节点翻译候选的类
class CandBeam
{
	public:
		~CandBeam() 
		{
			for (auto cand : cands)
			{
				delete cand;
			}
			for (auto cand : head_cands)
			{
				delete cand;
			}
		};
		void add(Cand *&cand_ptr,int beam_size);
		Cand* top() { return cands.front(); }
		Cand* at(size_t i) { return cands.at(i);}
		int size() { return cands.size();  }
		void sort() { std::sort(cands.begin(),cands.end(),larger); }
		void sort_head() { std::sort(head_cands.begin(),head_cands.end(),larger); }
	
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> cands;                         // 当前节点所有的翻译候选
		vector<Cand*> head_cands;                    // 当前节点由head rule生成的候选的翻译候选
};

typedef priority_queue<Cand*, vector<Cand*>, cmp> Candpq;

#endif
