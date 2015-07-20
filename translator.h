#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
#include "ruletable.h"
#include "syntaxtree.h"
#include "lm.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	LanguageModel *lm_model;
};

class SentenceTranslator
{
	public:
		SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen);
		~SentenceTranslator() {delete src_tree;};
		string translate_sentence();
		vector<TuneInfo> get_tune_info(size_t sen_id);
		vector<string> get_applied_rules(size_t sen_id);
	private:
        void fill_span2rules();
        void fill_span2cands_with_head_rule();
        void fill_span2rules_for_node(int node_idx);
		void generate_cand_with_head_rule(int node_idx);
        void generate_kbest_for_span(int beg, int span);
		vector<Rule> get_applicable_rules(int node_idx);
		void generalize_rule_src(SyntaxNode &node,string &config,vector<int> &generalized_rule_src, vector<int> &src_nt_idx_to_src_sen_idx);
		void generate_cand_with_rule_and_add_to_pq(Rule &rule,vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge,set<vector<int> > &duplicate_set);
		void generate_cand_with_glue_rule_and_add_to_pq(vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge,set<vector<int> > &duplicate_set);
		void add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge,set<vector<int> > &duplicate_set);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> &wids, int drop_oov);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		LanguageModel *lm_model;
		Parameter para;
		Weight feature_weight;

		vector<vector<CandBeam> > span2cands;		    //存储解码过程中所有跨度对应的候选列表, 
													    //span2cands[i][j]存储起始位置为i, 跨度为j的候选列表
		vector<vector<vector<Rule> > > span2rules;	    //存储每个跨度所有能用的hiero规则
        vector<vector<int> > span2head;                 //记录每个span对应的中心词节点的位置，如果该span没有对应的head-modifier结构，则为-1

		SyntaxTree* src_tree;
		size_t src_sen_len;
		set<string> open_tags;
        map<string,int> type2id;
		int src_nt_id;
		int tgt_nt_id;
		int tgt_null_id;
};
