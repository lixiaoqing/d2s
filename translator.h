#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
#include "ruletable.h"
#include "syntaxtree.h"
#include "lm.h"
#include "myutils.h"

struct RuleSrcUnit
{
	int type;
	string word;
	string tag;
	int idx;
};

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
		void translate_subtree(int sub_root_idx);
		void generate_kbest_for_node(int node_idx);
		vector<Rule> get_applicable_rules(int node_idx);
		void generate_cand_with_head_rule(int node_idx);
		bool generalize_rule_src(vector<RuleSrcUnit> &rule_src,string &config,vector<int> &generalized_rule_src, vector<int> &src_nt_idx_to_src_sen_idx);
		void generate_cand_with_rule_and_add_to_pq(Rule &rule,vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge);
		void generate_cand_with_glue_rule_and_add_to_pq(vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge);
		void add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge);
		bool is_config_valid(vector<RuleSrcUnit> &rule_src,string &config);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> &wids, int drop_oov);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		LanguageModel *lm_model;
		Parameter para;
		Weight feature_weight;

		SyntaxTree* src_tree;
		size_t src_sen_len;
		set<string> open_tags;
		int src_nt_id;
		int tgt_nt_id;
		int tgt_null_id;
};
