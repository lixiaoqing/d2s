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
	int btg_num_mono;				//生成当前候选所使用的正序btg规则数目
	int btg_num_swap;				//生成当前候选所使用的逆序btg规则数目

	//目标端信息
	int tgt_word_num;			//当前候选目标端的单词数
	vector<int> tgt_wids;		//当前候选译文的单词id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;
	double mono_prob;			//生成当前候选所使用的正序规则的调序概率总得分
	double swap_prob;			//生成当前候选所使用的逆序规则的调序概率总得分

	//来源信息, 记录候选是如何生成的
	Rule applied_rule;          					//生成当前候选所使用的规则
	vector<vector<Cand*> > cands_of_nt_leaves;      // 规则源端非终结符叶节点的翻译候选(btg规则所有叶节点均为非终结符)
                                                    // 注意排列顺序为规则目标端的非终结符顺序，btg规则除外
	vector<int> cand_rank_vec;                      // 记录当前候选所用的每个非终结符叶节点的翻译候选的排名
    Span src_span;                                  // 记录候选在源端的跨度
    int span_lhs;                                   // 由btg规则生成的候选在源端的第一个跨度的长度
    int sub_cand_order;                             // btg候选的合并顺序，0为顺序，1为逆序，-1为非btg候选

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		rule_num = 1;
		btg_num_mono = 0;
		btg_num_swap = 0;

		tgt_word_num = 1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.clear();
		lm_prob = 0.0;
		mono_prob = 0.0;
		swap_prob = 0.0;

		cands_of_nt_leaves.clear();
		cand_rank_vec.clear();
        src_span = make_pair(-1,-1);
        span_lhs = -1;
        sub_cand_order = -1;
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
		};
		void add(Cand *&cand_ptr,int beam_size);
		Cand* top() { return cands.front(); }
		Cand* at(size_t i) { return cands.at(i);}
		int size() { return cands.size();  }
		void sort() { std::sort(cands.begin(),cands.end(),larger); }
	
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> cands;                         // 当前节点所有的翻译候选
};

typedef priority_queue<Cand*, vector<Cand*>, cmp> Candpq;

#endif
