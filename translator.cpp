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

    span2cands.resize(src_sen_len);
    span2rules.resize(src_sen_len);
    span2head.resize(src_sen_len);
    for (int beg=0;beg<src_sen_len;beg++)
    {
        span2cands.at(beg).resize(src_sen_len-beg);
        span2rules.at(beg).resize(src_sen_len-beg);
        span2head.at(beg).resize(src_sen_len-beg,-2);
    }

    //为每个节点生成head-modifier规则
	fill_span2rules();
    //为每个节点生成head候选
	fill_span2cands_with_head_rule();
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
	vector<Cand*> &final_cands = span2cands.at(0).at(src_sen_len-1).cands;
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
	Cand *best_cand = span2cands.at(0).at(src_sen_len-1).cands[0];
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
        //rule += "0.0";
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
        //rule += to_string(applied_tgt_rule->score);
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

void SentenceTranslator::fill_span2cands_with_head_rule()
{
    for (int node_idx=0;node_idx<src_sen_len;node_idx++)
    {
        generate_cand_with_head_rule(node_idx);
    }
}

/**************************************************************************************
 1. 函数功能: 根据head规则为当前节点生成候选
 2. 入口参数: 当前节点的序号
 3. 出口参数: 无
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generate_cand_with_head_rule(int node_idx)
{
	auto &node = src_tree->nodes.at(node_idx);
	int src_wid = src_vocab->get_id(node.word);
	vector<int> src_wids = {src_wid};
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
		span2cands.at(node_idx).at(0).cands.push_back(cand);
	}
	else
	{
        for (int i=0;i<matched_rules->size();i++)
		{
            auto &tgt_rule = matched_rules->at(i);
			Cand* cand = new Cand;
			if (tgt_rule.wids.at(0) == tgt_null_id)                                     //deletion规则
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
			cand->applied_rule.tgt_rule_rank = i;
			cand->lm_prob = lm_model->cal_increased_lm_score(cand);
			cand->score += feature_weight.rule_num*cand->rule_num 
				         + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
            span2cands.at(node_idx).at(0).cands.push_back(cand);
        }
	}
    span2cands.at(node_idx).at(0).sort();
}

void SentenceTranslator::fill_span2rules()
{
    for (int node_idx=0;node_idx<src_sen_len;node_idx++)
    {
        fill_span2rules_for_node(node_idx);
    }
}

/**************************************************************************************
 1. 函数功能: 找到当前节点所有可用的fixed和floating规则
 2. 入口参数: 当前节点的序号
 3. 出口参数: 可用的规则列表
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::fill_span2rules_for_node(int node_idx)
{
	SyntaxNode &node = src_tree->nodes.at(node_idx);
    int children_size = node.children.size();
    for (int children_num=1;children_num<=children_size;children_num++)
    {
        for (int first_child_idx=0;first_child_idx<=children_size-children_num;first_child_idx++)
        {
            generate_rules(node,first_child_idx,children_num,"fixed");
            if (children_num > 1)
            {
                generate_rules(node,first_child_idx,children_num,"floating");
            }
        }
    }
}

void SentenceTranslator::generate_rules(SyntaxNode &node,int first_child_idx,int children_num,string struct_type)
{
    Span src_span = cal_src_span(node,first_child_idx,children_num,struct_type);
    if (src_span.first == -1)
        return;
	vector<vector<int> > generalized_rule_src_vec;                                  //记录泛化的规则源端
	vector<vector<int> > src_nt_idx_to_src_sen_idx_vec;                             //记录每个源端变量序号在句子中位置
    vector<string> configs = {"lll","llg","lgl","gll","lgg","glg","ggl","ggg"};
    for (string &config : configs)
    {
		vector<int> generalized_rule_src;
		vector<int> src_nt_idx_to_src_sen_idx;
        generalize_rule_src(node,first_child_idx,children_num,config,struct_type,generalized_rule_src,src_nt_idx_to_src_sen_idx);
		auto it = find(generalized_rule_src_vec.begin(),generalized_rule_src_vec.end(),generalized_rule_src);
		if (it == generalized_rule_src_vec.end())
		{
			generalized_rule_src_vec.push_back(generalized_rule_src);
			src_nt_idx_to_src_sen_idx_vec.push_back(src_nt_idx_to_src_sen_idx);
		}
    }

    //根据泛化的源端在规则表中寻找目标端
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
			rule.src_ids = generalized_rule_src;
			rule.tgt_rule = &(matched_rules->at(j));
			rule.tgt_rule_rank = j;
			//规则目标端变量位置到句子源端位置的映射
			for (int src_nt_idx : matched_rules->at(j).tgt_nt_idx_to_src_nt_idx)
			{
				rule.tgt_nt_idx_to_src_sen_idx.push_back(src_nt_idx_to_src_sen_idx[src_nt_idx]);
			}
			span2rules.at(src_span.first).at(src_span.second).push_back(rule);
		}
	}
}

/**************************************************************************************
 1. 函数功能: 对规则源端进行泛化
 2. 入口参数: 当前节点，泛化配置
 3. 出口参数: 泛化的规则源端，源端每个变量在句子中的位置
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generalize_rule_src(SyntaxNode &node,int first_child_idx,int children_num,string config,string struct_type,vector<int> &generalized_rule_src,vector<int> &src_nt_idx_to_src_sen_idx)
{
	for (int child_idx=first_child_idx; child_idx<first_child_idx+children_num;child_idx++)
	{
        SyntaxNode &child = src_tree->nodes.at(node.children.at(child_idx));        //注意此处child_idx表示当前节点是第几个孩子节点，
                                                                                    //而不是它在句子中的位置
		if (child.children.empty())													//叶节点
		{
			if (config[2] == 'g' && open_tags.find(child.tag) != open_tags.end() )
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+child.tag));
				src_nt_idx_to_src_sen_idx.push_back(child.idx);
			}
			else
			{
				generalized_rule_src.push_back(src_vocab->get_id(child.word));
			}
		}
		else	                              										//内部节点
		{
			if (config[1] == 'g')
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+child.tag));
			}
			else
			{
				generalized_rule_src.push_back(src_vocab->get_id("[x]"+child.word));
			}
			src_nt_idx_to_src_sen_idx.push_back(child.idx);
		}
    }
    //中心词节点
    if (struct_type == "fixed")
    {
        if (config[0] == 'g')
        {
            generalized_rule_src.push_back(src_vocab->get_id("[x]"+node.tag));
            src_nt_idx_to_src_sen_idx.push_back(node.idx);
        }
        else
        {
            generalized_rule_src.push_back(src_vocab->get_id(node.word));
        }
    }
}

Span SentenceTranslator::cal_src_span(SyntaxNode &node,int first_child_idx,int children_num,string struct_type)
{
	auto &first_child = src_tree->nodes.at(node.children.at(first_child_idx));
	auto &last_child = src_tree->nodes.at(node.children.at(first_child_idx+children_num-1));
    Span src_span;
    if (struct_type == "fixed")
    {
        //规则源端必须连续
        if (first_child.src_span.first-1 > node.idx || last_child.src_span.first+last_child.src_span.second+1 < node.idx)
            return make_pair(-1,-1);
        //首先合并第一个和最后一个孩子的源端span，然后与当前节点的源端span合并
        src_span = merge_span(merge_span(first_child.src_span,last_child.src_span),make_pair(node.idx,0));
        span2head.at(src_span.first).at(src_span.second) = node.idx;                    //记录当前span对应的中心词在句子中的位置
    }
    else
    {
        //规则源端必须连续
        if (first_child.idx < node.idx && last_child.idx > node.idx)
            return make_pair(-1,-1);
        //合并第一个和最后一个孩子的源端span
        Span src_span = merge_span(first_child.src_span,last_child.src_span);
        span2head.at(src_span.first).at(src_span.second) = -1;                          //将floating结构的中心词位置记为-1
    }
    return src_span;
}

string SentenceTranslator::translate_sentence()
{
	if (src_sen_len == 0)
		return "";
    for (int span=1;span<src_sen_len;span++)
    {
        for (int beg=0;beg+span<src_sen_len;beg++)
        {
            generate_kbest_for_span(beg,span);
        }
    }
	return words_to_str(span2cands.at(0).at(src_sen_len-1).cands[0]->tgt_wids,0);
}

/**************************************************************************************
 1. 函数功能: 为每个span生成kbest候选
 2. 入口参数: 当前span的起始位置和长度
 3. 出口参数: 无
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generate_kbest_for_span(int beg, int span)
{
    int head_idx = span2head.at(beg).at(span);
    if (head_idx == -2)
        return;
	Candpq candpq_merge;			                                                     //优先级队列,用来临时存储通过合并得到的候选
	set<vector<int> > duplicate_set;	                                                 //用来记录候选是否已经被加入candpq_merge中
	duplicate_set.clear();
	for(auto &rule : span2rules.at(beg).at(span))
	{
		vector<int> cand_rank_vec(rule.nt_num,0);
		vector<vector<Cand*> > cands_of_nt_leaves;
		for (int idx : rule.tgt_nt_idx_to_src_sen_idx)
		{
			if (idx != head_idx)
			{
                Span src_span = src_tree->nodes.at(idx).src_span;
				cands_of_nt_leaves.push_back(span2cands.at(src_span.first).at(src_span.second).cands);
			}
			else
			{
				cands_of_nt_leaves.push_back(span2cands.at(idx).at(0).cands);
			}
		}
		generate_cand_with_rule_and_add_to_pq(rule,cands_of_nt_leaves,cand_rank_vec,candpq_merge,duplicate_set);
	}
    for (int span_lhs=0;span_lhs<span;span_lhs++)       //使用btg规则生成候选
	{
        int beg_lhs = beg;
        int beg_rhs = beg_lhs+span_lhs+1;
        int span_rhs = span-span_lhs-1;
        if (span_lhs > 0 && span2head.at(beg_lhs).at(span_lhs) == -2)
            continue;
        if (span_rhs > 0 && span2head.at(beg_rhs).at(span_rhs) == -2)
            continue;
        vector<int> cand_rank_vec(2,0);
        vector<vector<Cand*> > cands_of_src_nt_leaves;
        cands_of_src_nt_leaves.push_back(span2cands.at(beg_lhs).at(span_lhs).cands);
        cands_of_src_nt_leaves.push_back(span2cands.at(beg_rhs).at(span_rhs).cands);
        generate_cand_with_btg_rule_and_add_to_pq(span_lhs,cands_of_src_nt_leaves,cand_rank_vec,candpq_merge,duplicate_set);
	}
	//立方体剪枝,每次从candpq_merge中取出最好的候选加入span2cands中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;
	while (added_cand_num<para.CUBE_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
		if (head_idx == src_tree->root_idx)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
        add_neighbours_to_pq(best_cand,candpq_merge,duplicate_set);
		span2cands.at(beg).at(span).add(best_cand,para.BEAM_SIZE);
		added_cand_num++;
	}
	while(!candpq_merge.empty())
	{
		delete candpq_merge.top();
		candpq_merge.pop();
	}
	span2cands.at(beg).at(span).sort();                                                     //对当前节点的候选进行排序
}

/**************************************************************************************
 1. 函数功能: 根据规则和规则目标端非终结符叶节点的翻译候选生成当前节点的候选
 2. 入口参数: 翻译规则，每个变量节点的所有翻译候选，每个变量节点被使用的候选的排名
              检查当前候选是否已经生成过的集合
 3. 出口参数: 保存当前节点候选的优先级队列
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::generate_cand_with_rule_and_add_to_pq(Rule &rule,vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge,set<vector<int> > &duplicate_set)
{
    //key包含规则中变量的个数，每个目标端变量在源端句子中对应的位置（用来检查规则源端是否相同）
    //规则目标端在源端相同的所有目标端的排名（检查规则目标端是否相同），以及子候选在每个个变量中的排名（检查子候选是否相同）
    vector<int> key;
    key.push_back(rule.nt_num);
    key.insert(key.end(),rule.tgt_nt_idx_to_src_sen_idx.begin(),rule.tgt_nt_idx_to_src_sen_idx.end());
    key.push_back(rule.tgt_rule_rank);
    key.insert(key.end(),cand_rank_vec.begin(),cand_rank_vec.end());
    if (duplicate_set.insert(key).second == false)
        return;

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
}

/**************************************************************************************
 1. 函数功能: 根据btg规则和当前跨度的两个子跨度上的翻译候选生成当前跨度的候选
 2. 入口参数: 每个子跨度的翻译候选列表，使用的候选在所在列表中的排名
              检查当前候选是否已经生成过的集合
 3. 出口参数: 保存当前节点候选的优先级队列
 4. 算法简介: 将两个子跨度的候选顺序和逆序拼接得到当前跨度的候选
***************************************************************************************/
void SentenceTranslator::generate_cand_with_btg_rule_and_add_to_pq(int span_lhs,vector<vector<Cand*> > &cands_of_src_nt_leaves, vector<int> &cand_rank_vec,Candpq &candpq_merge,set<vector<int> > &duplicate_set)
{
    //key包含左span的长度（检查规则是否相同），以及子候选在每个变量中的排名（检查子候选是否相同）
    vector<int> key;
    key.push_back(span_lhs);
    key.insert(key.end(),cand_rank_vec.begin(),cand_rank_vec.end());
    if (duplicate_set.insert(key).second == false)
        return;

    vector<string> nt_orders = {"mono","swap"};
    for (string &order : nt_orders)
    {
        Cand *cand = new Cand;
        int nt_num = cands_of_src_nt_leaves.size();
        vector<int> src_ids(nt_num,src_nt_id);
        vector<int> tgt_nt_idx_to_src_sen_idx(nt_num,0);
        Rule glue_rule = {nt_num,src_ids,NULL,0,tgt_nt_idx_to_src_sen_idx};
        cand->applied_rule = glue_rule;
        cand->cands_of_nt_leaves = cands_of_src_nt_leaves;
        vector<vector<Cand*> > cands_of_tgt_nt_leaves = cands_of_src_nt_leaves;
        cand->cand_rank_vec = cand_rank_vec;
        cand->sub_cand_order = 0;
        if (order == "swap")
        {
            reverse(cands_of_tgt_nt_leaves.begin(),cands_of_tgt_nt_leaves.end());
            reverse(cand_rank_vec.begin(),cand_rank_vec.end());
            cand->sub_cand_order = 1;
        }
        cand->span_lhs = span_lhs;

        cand->trans_probs.resize(PROB_NUM,0.0);                                                              // 初始化当前候选的翻译概率
        for (int nt_idx=0;nt_idx<nt_num;nt_idx++)
        {
            Cand* subcand = cands_of_tgt_nt_leaves[nt_idx][cand_rank_vec[nt_idx]];
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
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq中
 2. 入口参数: 当前候选, 检查是否重复扩展的duplicate_set
 3. 出口参数: 更新后的candpq
 4. 算法简介: a) 对于btg规则生成的候选, 考虑它所有非终结符叶节点的下一位候选
              b) 对于普通规则生成的候选, 考虑叶节点候选的下一位以及规则的下一位
***************************************************************************************/
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge, set<vector<int> > &duplicate_set)
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
				generate_cand_with_rule_and_add_to_pq(cur_cand->applied_rule,cur_cand->cands_of_nt_leaves,new_cand_rank_vec,candpq_merge,duplicate_set);
			}
			else                                                      // glue规则生成的候选
			{
				generate_cand_with_btg_rule_and_add_to_pq(cur_cand->span_lhs,cur_cand->cands_of_nt_leaves,new_cand_rank_vec,candpq_merge,duplicate_set);
			}
		}
	}
}

