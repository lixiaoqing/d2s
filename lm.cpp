#include "lm.h"

struct ID_converter : public lm::EnumerateVocab 
{
	ID_converter(vector<lm::WordIndex>* out, Vocab* vocab) : sub_to_kenlm_id(out), UNK_ID(0),tgt_vocab(vocab) { sub_to_kenlm_id->clear(); }
	void Add(lm::WordIndex index, const StringPiece &str) 
	{
		const int ori_id = tgt_vocab->get_id(str.as_string());
		if (ori_id >= sub_to_kenlm_id->size())
		{
			sub_to_kenlm_id->resize(ori_id + 1, UNK_ID);
		}
		sub_to_kenlm_id->at(ori_id) = index;
	}
	vector<lm::WordIndex>* sub_to_kenlm_id;
	const lm::WordIndex UNK_ID;
	Vocab* tgt_vocab;
};

LanguageModel::LanguageModel(const string &lm_file, Vocab *tgt_vocab)
{
	ID_converter id_converter(&ori_to_kenlm_id,tgt_vocab);
	Config conf;
	conf.enumerate_vocab = &id_converter;
	kenlm = new Model(lm_file.c_str(), conf);
	EOS = convert_to_kenlm_id(tgt_vocab->get_id("</s>"));
	null_klm_id = convert_to_kenlm_id(tgt_vocab->get_id("NULL"));
	nonterminal_wid = tgt_vocab->get_id("[x]");
	cout<<"load language model file "<<lm_file<<" over\n";
};

lm::WordIndex LanguageModel::convert_to_kenlm_id(int wid)
{
	if (wid >= ori_to_kenlm_id.size())
		return 0;
	else
		return ori_to_kenlm_id[wid];
}

double LanguageModel::cal_increased_lm_score(Cand* cand) 
{
	RuleScore<Model> rule_score(*kenlm,cand->lm_state);
	if (cand->applied_rule.tgt_rule == NULL)
	{
		if (cand->applied_rule.nt_num == 0)            //OOV候选
		{
			rule_score.Terminal(null_klm_id);
		}
		else
		{
			for (int nt_idx=0; nt_idx<cand->applied_rule.nt_num;nt_idx++)
			{
				rule_score.NonTerminal(cand->cands_of_nt_leaves.at(nt_idx).at(cand->cand_rank_vec.at(nt_idx))->lm_state);
			}
		}
	}
	else
	{
		size_t nt_idx = 0;
		for (auto wid : cand->applied_rule.tgt_rule->wids)
		{
			if (wid == nonterminal_wid)
			{
				rule_score.NonTerminal(cand->cands_of_nt_leaves.at(nt_idx).at(cand->cand_rank_vec.at(nt_idx))->lm_state);
				nt_idx++;
			}
			else
			{
				const lm::WordIndex ken_lm_id = convert_to_kenlm_id(wid);
				rule_score.Terminal(ken_lm_id);
			}
		}
	}
	double increased_lm_score = rule_score.Finish();
	cand->lm_state.ZeroRemaining();
	return increased_lm_score;
}

double LanguageModel::cal_final_increased_lm_score(Cand* cand) 
{
	ChartState cstate;
	RuleScore<Model> rule_score(*kenlm, cstate);
	rule_score.BeginSentence();
	rule_score.NonTerminal(cand->lm_state, 0.0f);
	rule_score.Terminal(EOS);
	return rule_score.Finish();
}
