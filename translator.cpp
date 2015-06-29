#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	open_tags = {"CD","OD","DT","JJ","NN","NR","NT","AD","FW","PN"};
    type2id = {{"lll",0},{"gll",1},{"lgl",2},{"llg",3},{"lgg",4},{"glg",5},{"ggl",6},{"ggg",7}};
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	lm_model = i_models.lm_model;
	para = i_para;
	feature_weight = i_weight;

	src_tree = new SyntaxTree(input_sen);
	src_sen_len = src_tree->sen_len;
	src_nt_id = src_vocab->get_id("[x]");
	tgt_nt_id = tgt_vocab->get_id("[x]");
	tgt_null_id = tgt_vocab->get_id("NULL");
}

string SentenceTranslator::words_to_str(vector<int> &wids, int drop_oov)
{
		string output = "";
		for (const auto &wid : wids)
		{
			if (wid >= 0)
			{
				output += tgt_vocab->get_word(wid) + " ";
			}
			else if (drop_oov == 0)
			{
				output += src_vocab->get_word(0-wid) + " ";
			}
		}
		TrimLine(output);
		return output;
}

vector<TuneInfo> SentenceTranslator::get_tune_info(size_t sen_id)
{
	vector<TuneInfo> nbest_tune_info;
	vector<Cand*> &final_cands = src_tree->nodes.at(src_tree->root_idx).cand_organizer.cands;
	for (size_t i=0;i< min(final_cands.size(),(size_t)para.NBEST_NUM);i++)
	{
		TuneInfo tune_info;
		tune_info.sen_id = sen_id;
		tune_info.translation = words_to_str(final_cands.at(i)->tgt_wids,0);
		for (size_t j=0;j<PROB_NUM;j++)
		{
			tune_info.feature_values.push_back(final_cands.at(i)->trans_probs.at(j));
		}
		tune_info.feature_values.push_back(final_cands.at(i)->lm_prob);
		tune_info.feature_values.push_back(final_cands.at(i)->tgt_wids.size() );
		tune_info.feature_values.push_back(final_cands.at(i)->rule_num);
		tune_info.feature_values.push_back(final_cands.at(i)->glue_num);
		tune_info.total_score = final_cands.at(i)->score;
		nbest_tune_info.push_back(tune_info);
	}
	return nbest_tune_info;
}

vector<string> SentenceTranslator::get_applied_rules(size_t sen_id)
{
	vector<string> applied_rules;
	Cand *best_cand = src_tree->nodes.at(src_tree->root_idx).cand_organizer.cands[0];
	dump_rules(applied_rules,best_cand);
	applied_rules.push_back(" ||||| ");
	string src_sen;
	for (auto &node : src_tree->nodes)
	{
		src_sen += node.word + " ";
	}
	applied_rules.push_back(src_sen);
	return applied_rules;
}

/**************************************************************************************
 1. 函数功能: 获取当前候选所使用的规则
 2. 入口参数: 当前候选的指针
 3. 出口参数: 用于记录规则的applied_rules
 4. 算法简介: 通过递归的方式自顶向下回溯, 查找所使用的规则
***************************************************************************************/
void SentenceTranslator::dump_rules(vector<string> &applied_rules, Cand *cand)
{
	applied_rules.push_back(" ");
	string rule;
	int nt_num = cand->applied_rule.nt_num;
	if (nt_num > 0)
	{
		applied_rules.push_back(" ( ");
	}
	int nt_idx = 0;
	for (auto src_wid : cand->applied_rule.src_ids)
	{
		string src_unit = src_vocab->get_word(src_wid)+"_";
		if (src_unit.substr(0,3) == "[x]")
		{
			rule += "[x"+to_string(nt_idx)+"]"+src_unit.substr(3);
			nt_idx++;
		}
		else
		{
			rule += src_unit;
		}
	}

	rule += "|||_";
	if (cand->applied_rule.tgt_rule == NULL)
	{
		if (nt_num == 0)
		{
			rule += "OOV_";
		}
		else
		{
			for (int i=0; i<nt_num; i++)
			{
				rule += "[x"+to_string(i)+"]_";
			}
		}
	}
	else
	{
		nt_idx = 0;
		auto &applied_tgt_rule = cand->applied_rule.tgt_rule;
		for (auto tgt_wid : applied_tgt_rule->wids)
		{
			if (tgt_wid == tgt_nt_id)
			{
				rule += "[x"+to_string(applied_tgt_rule->tgt_nt_idx_to_src_nt_idx.at(nt_idx))+"]_";
				nt_idx++;
			}
			else
			{
				rule += tgt_vocab->get_word(tgt_wid)+"_";
			}
		}
	}
	rule.erase(rule.end()-1);
	applied_rules.push_back(rule);
	if (nt_num > 0)
	{
		auto &applied_tgt_rule = cand->applied_rule.tgt_rule;
		if (applied_tgt_rule == NULL)
		{
		   for (int i=0;i<nt_num;i++)
		   {
			   dump_rules(applied_rules,cand->cands_of_nt_leaves.at(i).at(cand->cand_rank_vec.at(i)));
		   }
		}
		else
		{
			vector<int> src_nt_idx_to_tgt_nt_idx(nt_num,0);
			for (int i=0; i<nt_num; i++)
			{
				src_nt_idx_to_tgt_nt_idx.at(applied_tgt_rule->tgt_nt_idx_to_src_nt_idx.at(i)) = i;
			}
			for (int i=0;i<nt_num;i++)
			{
				int j = src_nt_idx_to_tgt_nt_idx.at(i);
				dump_rules(applied_rules,cand->cands_of_nt_leaves.at(j).at(cand->cand_rank_vec.at(j)));
			}
		}
		applied_rules.push_back(" ) ");
	}
}

string SentenceTranslator::translate_sentence()
{
	if (src_sen_len == 0)
		return "";
	translate_subtree(src_tree->root_idx);
	return words_to_str(src_tree->nodes.at(src_tree->root_idx).cand_organizer.cands[0]->tgt_wids,0);
}

void SentenceTranslator::translate_subtree(int sub_root_idx)
{
	for (int idx : src_tree->nodes.at(sub_root_idx).children)
	{
		translate_subtree(idx);
	}
	generate_kbest_for_node(sub_root_idx);
}

/**************************************************************************************
 1. 函数功能: 为每个句法树节点生成kbest候选
 2. 入口参数: 指向句法树节点的指针
 3. 出口参数: 无
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generate_kbest_for_node(int node_idx)
{
	SyntaxNode &node = src_tree->nodes.at(node_idx);
	generate_cand_with_head_rule(node_idx);
	//cout<<"generate cand with head rule for node "<<src_tree->nodes.at(node_idx).word<<endl;
	if ( node.children.empty() )                                                          // 叶节点
	{
		swap(node.cand_organizer.cands, node.cand_organizer.head_cands);
		node.cand_organizer.sort();
		//cout<<"generate cand for leaf node "<<src_tree->nodes.at(node_idx).word<<endl;
		return;
	}

	vector<Rule> applicable_rules = get_applicable_rules(node_idx);
	Candpq candpq_merge;			//优先级队列,用来临时存储通过合并得到的候选
	for(auto &rule : applicable_rules)
	{
		vector<int> cand_rank_vec(rule.nt_num,0);
		vector<vector<Cand*> > cands_of_nt_leaves;
		for (int idx : rule.tgt_nt_idx_to_src_sen_idx)
		{
			if (idx != node_idx)
			{
				cands_of_nt_leaves.push_back(src_tree->nodes.at(idx).cand_organizer.cands);
			}
			else
			{
				cands_of_nt_leaves.push_back(src_tree->nodes.at(idx).cand_organizer.head_cands);
			}
			//cout<<src_tree->nodes.at(idx).word<<" "<<cands_of_nt_leaves.back().size()<<endl;
		}
		generate_cand_with_rule_and_add_to_pq(rule,cands_of_nt_leaves,cand_rank_vec,candpq_merge);
		//cout<<"generate cand with head-modifier rule for node "<<src_tree->nodes.at(node_idx).word<<endl;
	}
	if (candpq_merge.empty())       //使用glue规则生成候选
	{
		int nt_num = node.children.size() + 1;
		vector<int> cand_rank_vec(nt_num,0);
		vector<vector<Cand*> > cands_of_nt_leaves;
		bool flag = false;						//检查中心词节点的候选是否已经被加入
		for (int idx : node.children)
		{
			if (idx > node_idx && flag == false)
			{
				cands_of_nt_leaves.push_back(src_tree->nodes.at(node_idx).cand_organizer.head_cands);
				flag = true;
			}
			cands_of_nt_leaves.push_back(src_tree->nodes.at(idx).cand_organizer.cands);
		}
		if (flag == false)
		{
			cands_of_nt_leaves.push_back(src_tree->nodes.at(node_idx).cand_organizer.head_cands);
		}

		generate_cand_with_glue_rule_and_add_to_pq(cands_of_nt_leaves,cand_rank_vec,candpq_merge);
		//cout<<"generate cand with glue rule for node "<<src_tree->nodes.at(node_idx).word<<endl;
	}
	set<vector<int> > duplicate_set;	//用来记录candpq_merge中的候选是否已经被扩展过
	duplicate_set.clear();
	//立方体剪枝,每次从candpq_merge中取出最好的候选加入span2cands中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;
	while (added_cand_num<para.CUBE_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
		if (node_idx == src_tree->root_idx)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		
		//key包含规则中变量的个数，每个目标端变量在源端句子中对应的位置，子候选在每个变量中的排名，以及规则目标端在源端相同的所有目标端的排名
		vector<int> key;
		key.push_back(best_cand->applied_rule.nt_num);
		key.insert(key.end(),best_cand->applied_rule.tgt_nt_idx_to_src_sen_idx.begin(),best_cand->applied_rule.tgt_nt_idx_to_src_sen_idx.end());
		key.insert(key.end(),best_cand->cand_rank_vec.begin(),best_cand->cand_rank_vec.end());
		key.push_back(best_cand->applied_rule.tgt_rule_rank);
		if (duplicate_set.find(key) == duplicate_set.end())
		{
			add_neighbours_to_pq(best_cand,candpq_merge);
			duplicate_set.insert(key);
		}
		node.cand_organizer.add(best_cand,para.BEAM_SIZE);
		added_cand_num++;
	}
	while(!candpq_merge.empty())
	{
		delete candpq_merge.top();
		candpq_merge.pop();
	}
	node.cand_organizer.sort();
	//cout<<"cube prunning over\n";
}

vector<Rule> SentenceTranslator::get_applicable_rules(int node_idx)
{
	SyntaxNode &node = src_tree->nodes.at(node_idx);
	vector<RuleSrcUnit> rule_src;
	for (auto child_idx : node.children)
	{
		auto &child = src_tree->nodes.at(child_idx);
		if (child.children.empty())
		{
			RuleSrcUnit unit = {2,child.word,child.tag,child.idx};
			rule_src.push_back(unit);
		}
		else
		{
			RuleSrcUnit unit = {1,child.word,child.tag,child.idx};
			rule_src.push_back(unit);
		}
	}
	RuleSrcUnit unit = {0,node.word,node.tag,node.idx};
	rule_src.push_back(unit);

	vector<vector<int> > generalized_rule_src_vec;
	vector<vector<int> > src_nt_idx_to_src_sen_idx_vec;
	vector<string> configs = {"lll","llg","lgl","gll","lgg","glg","ggl","ggg"};
	for (string &config : configs)
	{
		vector<int> generalized_rule_src = {type2id[config]};
		vector<int> src_nt_idx_to_src_sen_idx;
		bool flag = generalize_rule_src(rule_src,config,generalized_rule_src,src_nt_idx_to_src_sen_idx);
		if (flag == false)
			continue;
		auto it = find(generalized_rule_src_vec.begin(),generalized_rule_src_vec.end(),generalized_rule_src);
		if (it == generalized_rule_src_vec.end())
		{
			generalized_rule_src_vec.push_back(generalized_rule_src);
			src_nt_idx_to_src_sen_idx_vec.push_back(src_nt_idx_to_src_sen_idx);
		}
	}
	//cout<<"genearlize rule src over\n";

	vector<Rule> applicable_rules;
	for (int i=0; i<generalized_rule_src_vec.size(); i++)
	{
		auto &generalized_rule_src = generalized_rule_src_vec.at(i);
		auto &src_nt_idx_to_src_sen_idx = src_nt_idx_to_src_sen_idx_vec.at(i);
		vector<TgtRule>* matched_rules = ruletable->find_matched_rules(generalized_rule_src);
		if (matched_rules == NULL)
			continue;
		for (int j=0;j<matched_rules->size();j++)
		{
			Rule rule;
			rule.nt_num = src_nt_idx_to_src_sen_idx.size();
            vector<int> src_ids(generalized_rule_src.begin()+1,generalized_rule_src.end());
			rule.src_ids = src_ids;
			rule.tgt_rule = &(matched_rules->at(j));
			rule.tgt_rule_rank = j;
			//规则目标端变量位置到句子源端位置的映射
			for (int src_nt_idx : matched_rules->at(j).tgt_nt_idx_to_src_nt_idx)
			{
				rule.tgt_nt_idx_to_src_sen_idx.push_back(src_nt_idx_to_src_sen_idx[src_nt_idx]);
			}
			applicable_rules.push_back(rule);
		}
	}
	//cout<<"get applicable rules over\n";
	return applicable_rules;
}

void SentenceTranslator::generate_cand_with_head_rule(int node_idx)
{
	auto &node = src_tree->nodes.at(node_idx);
	int src_wid = src_vocab->get_id(node.word);
	vector<int> src_wids = {type2id["lll"],src_wid};
	vector<TgtRule>* matched_rules = ruletable->find_matched_rules(src_wids);
	if (matched_rules == NULL)															//OOV
	{
		Cand* cand = new Cand;
		cand->tgt_wids.push_back(0 - src_wid);                                          //源端单词id取反，输出译文时使用
		cand->trans_probs.resize(PROB_NUM,0.0);
		cand->applied_rule.nt_num = 0;
		cand->applied_rule.src_ids.push_back(src_wid);
		cand->applied_rule.tgt_rule = NULL;
		cand->applied_rule.tgt_rule_rank = 0;
		cand->lm_prob = lm_model->cal_increased_lm_score(cand);
		cand->score += feature_weight.rule_num*cand->rule_num 
					+ feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
		node.cand_organizer.head_cands.push_back(cand);
	}
	else
	{
		for (auto &tgt_rule : *matched_rules)
		{
			Cand* cand = new Cand;
			if (tgt_rule.word_num == 1 && tgt_rule.wids.at(0) == tgt_null_id)           //deletion规则
			{
				cand->tgt_word_num = 0;
			}
			else
			{
				cand->tgt_word_num = tgt_rule.word_num;
				cand->tgt_wids = tgt_rule.wids;
			}
			cand->trans_probs = tgt_rule.probs;
			cand->score = tgt_rule.score;
			cand->applied_rule.nt_num = 0;
			cand->applied_rule.src_ids.push_back(src_wid);
			cand->applied_rule.tgt_rule = &tgt_rule;
			cand->applied_rule.tgt_rule_rank = 0;
			cand->lm_prob = lm_model->cal_increased_lm_score(cand);
			cand->score += feature_weight.rule_num*cand->rule_num 
				         + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
			node.cand_organizer.head_cands.push_back(cand);
		}
	}
}

bool SentenceTranslator::generalize_rule_src(vector<RuleSrcUnit> &rule_src,string &config,vector<int> &generalized_rule_src, vector<int> &src_nt_idx_to_src_sen_idx)
{
	if (is_config_valid(rule_src,config) == false)
		return false;
	for (auto &unit : rule_src)
	{
		if (unit.type == 2)													//叶节点
		{
			if (config[2] == 'g' && open_tags.find(unit.tag) != open_tags.end() )
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+unit.tag));
				src_nt_idx_to_src_sen_idx.push_back(unit.idx);
			}
			else
			{
				generalized_rule_src.push_back(src_vocab->get_id(unit.word));
			}
		}
		else if (unit.type == 1)											//内部节点
		{
			if (config[1] == 'g')
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+unit.tag));
			}
			else
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+unit.word));
			}
			src_nt_idx_to_src_sen_idx.push_back(unit.idx);
		}
		else if (unit.type == 0)											//中心词节点
		{
			if (config[1] == 'g')
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+unit.tag));
				src_nt_idx_to_src_sen_idx.push_back(unit.idx);
			}
			else
			{
				generalized_rule_src.push_back(src_vocab->get_id(unit.word));
			}
		}
	}
	return true;
}

bool SentenceTranslator::is_config_valid(vector<RuleSrcUnit> &rule_src,string &config)
{
    for (auto &unit : rule_src)
    {
        if (unit.type == 2 && config[2] == 'g' && open_tags.find(unit.tag) == open_tags.end())
            return false;
    }
    return true;
}

/**************************************************************************************
 1. 函数功能: 根据规则和规则目标端非终结符叶节点的翻译候选生成当前节点的候选
 2. 入口参数: a) 非终结符叶节点相同的规则列表 b) 使用的规则在规则列表中的排名
              c) 每个非终结符叶节点的翻译候选 d) 使用的每个翻译候选在它所在列表中的排名
 3. 出口参数: 指向新生成的候选的指针
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generate_cand_with_rule_and_add_to_pq(Rule &rule,vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge)
{
	Cand *cand = new Cand;
	cand->applied_rule = rule;
	cand->cands_of_nt_leaves = cands_of_nt_leaves;
	cand->cand_rank_vec = cand_rank_vec;
	
	cand->trans_probs = rule.tgt_rule->probs;                                             // 初始化当前候选的翻译概率
	size_t nt_idx = 0;
	for (int tgt_wid : rule.tgt_rule->wids)
	{
		if (tgt_wid != tgt_nt_id)
		{
			cand->tgt_wids.push_back(tgt_wid);                                            // 将规则目标端的词加入当前候选的译文
		}
		else
		{
			Cand* subcand = cands_of_nt_leaves[nt_idx][cand_rank_vec[nt_idx]];
			cand->tgt_wids.insert( cand->tgt_wids.end(),subcand->tgt_wids.begin(),subcand->tgt_wids.end() ); // 加入规则目标端非终结符的译文
			cand->rule_num += subcand->rule_num;                                                             // 累加所用的规则数量
			cand->glue_num += subcand->glue_num;                                                             // 累加所用的glue规则数量
			for (size_t j=0; j<PROB_NUM; j++)
			{
				cand->trans_probs[j] += subcand->trans_probs[j];                                             // 累加翻译概率
			}
			cand->lm_prob += subcand->lm_prob;                                                               // 累加语言模型得分
			cand->score   += subcand->score;                                                                 // 累加候选得分
			nt_idx++;
		}
	}
	double increased_lm_score = lm_model->cal_increased_lm_score(cand);                                      // 计算语言模型增量
	cand->lm_prob += increased_lm_score;
	cand->score   += rule.tgt_rule->score + feature_weight.lm*increased_lm_score + feature_weight.len*rule.tgt_rule->word_num
                     + feature_weight.rule_num*1;
	candpq_merge.push(cand);
	/*
	for (int wid : rule.src_ids)
		cout<<src_vocab->get_word(wid)<<' ';
	cout<<"||| ";
	for (int wid : rule.tgt_rule->wids)
		cout<<tgt_vocab->get_word(wid)<<' ';
	cout<<"||| ";
	cout<<words_to_str(cand->tgt_wids,0);
	cout<<endl;
	*/
}

/**************************************************************************************
 1. 函数功能: 根据glue规则和当前句法节点的所有子节点的翻译候选生成当前节点的候选
 2. 入口参数: a) 当前句法节点的每个子节点的翻译候选列表 b) 使用的候选在所在列表中的排名
 3. 出口参数: 指向新生成的候选的指针
 4. 算法简介: 将当前句法节点的所有子节点的翻译候选顺序拼接即可
***************************************************************************************/
void SentenceTranslator::generate_cand_with_glue_rule_and_add_to_pq(vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge)
{
	Cand *cand = new Cand;
	int nt_num = cands_of_nt_leaves.size();
	vector<int> src_ids(nt_num,src_nt_id);
	vector<int> tgt_nt_idx_to_src_sen_idx(nt_num,0);
	Rule glue_rule = {nt_num,src_ids,NULL,0,tgt_nt_idx_to_src_sen_idx};
	cand->applied_rule = glue_rule;
	cand->cands_of_nt_leaves = cands_of_nt_leaves;
	cand->cand_rank_vec = cand_rank_vec;
	
	cand->trans_probs.resize(PROB_NUM,0.0);                                                              // 初始化当前候选的翻译概率
	for (int nt_idx=0;nt_idx<nt_num;nt_idx++)
	{
		Cand* subcand = cands_of_nt_leaves[nt_idx][cand_rank_vec[nt_idx]];
		cand->tgt_wids.insert( cand->tgt_wids.end(),subcand->tgt_wids.begin(),subcand->tgt_wids.end() ); // 加入规则目标端非终结符的译文
		cand->rule_num += subcand->rule_num;                                                             // 累加所用的规则数量
		cand->glue_num += subcand->glue_num;                                                             // 累加所用的glue规则数量
		for (size_t j=0; j<PROB_NUM; j++)
		{
			cand->trans_probs[j] += subcand->trans_probs[j];                                             // 累加翻译概率
		}
		cand->lm_prob += subcand->lm_prob;                                                               // 累加语言模型得分
		cand->score   += subcand->score;                                                                 // 累加候选得分
	}
	double increased_lm_score = lm_model->cal_increased_lm_score(cand);                                      // 计算语言模型增量
	cand->glue_num += 1;
	cand->lm_prob  += increased_lm_score;
	cand->score    += feature_weight.lm*increased_lm_score + feature_weight.glue*1;
	candpq_merge.push(cand);
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq中
 2. 入口参数: 当前候选, 检查是否重复扩展的duplicate_set
 3. 出口参数: 更新后的candpq
 4. 算法简介: a) 对于glue规则生成的候选, 考虑它所有非终结符叶节点的下一位候选
              b) 对于普通规则生成的候选, 考虑叶节点候选的下一位以及规则的下一位
***************************************************************************************/
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge)
{
    // 遍历所有非终结符叶节点, 若候选所用规则目标端无非终结符则不会进入此循环
	for (size_t i=0; i<cur_cand->cands_of_nt_leaves.size(); i++)
	{
		if ( cur_cand->cand_rank_vec[i]+1 < cur_cand->cands_of_nt_leaves[i].size() )
		{
			vector<int> new_cand_rank_vec = cur_cand->cand_rank_vec;
			new_cand_rank_vec[i]++;                                   // 考虑当前非终结符叶节点候选的下一位
			if (cur_cand->applied_rule.tgt_rule != NULL)              // 普通规则生成的候选
			{
				generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,cur_cand->cands_of_nt_leaves,new_cand_rank_vec,candpq_merge);
			}
			else         // glue规则生成的候选
			{
				generate_cand_with_glue_rule_and_add_to_pq(cur_cand->cands_of_nt_leaves,new_cand_rank_vec,candpq_merge);
			}
		}
	}
}

