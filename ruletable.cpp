#include "ruletable.h"

void RuleTable::load_rule_table(const string &rule_table_file)
{
	ifstream fin(rule_table_file.c_str(),ios::binary);
	if (!fin.is_open())
	{
		cerr<<"cannot open rule table file!\n";
		return;
	}
	int src_rule_len=0;
	while(fin.read((char*)&src_rule_len,sizeof(int)))
	{
		vector<int> src_wids;
		src_wids.resize(src_rule_len);
		fin.read((char*)&src_wids[0],sizeof(int)*src_rule_len);

		int tgt_rule_len=0;
		fin.read((char*)&tgt_rule_len,sizeof(int));
		TgtRule tgt_rule;
		tgt_rule.word_num = tgt_rule_len;
		tgt_rule.wids.resize(tgt_rule_len);
		fin.read((char*)&(tgt_rule.wids[0]),sizeof(int)*tgt_rule_len);

		int nt_num;
		fin.read((char*)&nt_num,sizeof(int));
		tgt_rule.nt_num = nt_num;
		if (nt_num > 0)
		{
			tgt_rule.tgt_nt_idx_to_src_nt_idx.resize(nt_num);
			fin.read((char*)&(tgt_rule.tgt_nt_idx_to_src_nt_idx[0]),sizeof(int)*nt_num);
		}

		tgt_rule.probs.resize(PROB_NUM);
		fin.read((char*)&(tgt_rule.probs[0]),sizeof(double)*PROB_NUM);

		tgt_rule.score = 0;
		if( tgt_rule.probs.size() != weight.trans.size() )
		{
			cout<<"number of probability in rule is wrong!"<<endl;
		}
		for( size_t i=0; i<weight.trans.size(); i++ )
		{
			tgt_rule.score += tgt_rule.probs[i]*weight.trans[i];
		}

		add_rule_to_trie(src_wids,tgt_rule);

		/*
		for (int src_wid : src_wids)
		{
			cout<<src_vocab->get_word(src_wid)<<" ";
		}
		cout<<"||| ";
		for (int tgt_wid : tgt_rule.wids)
		{
			cout<<tgt_vocab->get_word(tgt_wid)<<" ";
		}
		cout<<"||| ";
		cout<<tgt_rule.tgt_nt_idx_to_src_nt_idx.size()<<" ";
		for (int src_nt_idx : tgt_rule.tgt_nt_idx_to_src_nt_idx)
		{
			cout<<src_nt_idx<<" ";
		}
		cout<<"||| ";
		for (double prob : tgt_rule.probs)
		{
			cout<<prob<<" ";
		}
		cout<<endl;
		*/
	}
	fin.close();
	cout<<"load rule table file "<<rule_table_file<<" over\n";
}

vector<TgtRule>* RuleTable::find_matched_rules(const vector<int> &src_wids)
{
	RuleTrieNode* current = root;
	for (auto wid : src_wids)
	{
		auto it = current->id2subtrie_map.find(wid);
		if (it != current->id2subtrie_map.end())
		{
			current = it->second;
		}
		else
			return NULL;
	}
	return &(current->tgt_rules);
}

void RuleTable::add_rule_to_trie(const vector<int> &src_wids, const TgtRule &tgt_rule)
{
	RuleTrieNode* current = root;
	for (const auto &wid : src_wids)
	{        
		auto it = current->id2subtrie_map.find(wid);
		if ( it != current->id2subtrie_map.end() )
		{
			current = it->second;
		}
		else
		{
			RuleTrieNode* tmp = new RuleTrieNode();
			current->id2subtrie_map.insert(make_pair(wid,tmp));
			current = tmp;
		}
	}
	if (current->tgt_rules.size() < RULE_NUM_LIMIT)
	{
		current->tgt_rules.push_back(tgt_rule);
	}
	else
	{
		auto it = min_element(current->tgt_rules.begin(), current->tgt_rules.end());
		if( it->score < tgt_rule.score )
		{
			(*it) = tgt_rule;
		}
	}
}
