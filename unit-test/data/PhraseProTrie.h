/*
	PhraseProTrie.h
	phrase翻译概率的存储结构
*/
#ifndef PHRASEPROTRIE_H
#define PHRASEPROTRIE_H
#include "StdAfx.h"
#include "Vocab.h"
#include <vector>
#include <string>

using namespace std;

//存储英文短语
class s_PhrasePro 
{	
public:
	bool operator<(const s_PhrasePro &right) const;
	s_PhrasePro& operator=(const s_PhrasePro& right);
	s_PhrasePro()
	{
		ulEnNum = 0;
		viEnPhrase.clear();
		NodeIndexVec.clear();
		composedFlag = 0;
		LexFlag = 0;
		rootIndex = -1;
		dPro = 0.0;
		eachTransPro.clear();
		ngramPro = 0.0;
		strRule = "";
		SrcRule.clear();
	}
	int ulEnNum;						//英文词的个数

	//由叶节点序列，即viEnPhrase和NodeIndexVec组成的规则标识符
	string strRule;   
	vector<int> NodeIndexVec;
	vector<int> viEnPhrase;  //若是没有变量的翻译规则，那么英文就放在该结构中
	int composedFlag;     //记录该规则是最小规则还是组合规则（minimal or composed）
	int LexFlag;          //记录该规则是完全词汇化规则还是非词汇化规则
	int rootIndex;                //规则的根节点
	double dPro;					//概率

   //短语翻译中只有4个概率，s2tree中有5个概率（多一个规则的归约概率）, t2t有几个待定
	vector<double> eachTransPro;
	double ngramPro;           //若该规则的语言模型计算过则保留

	vector<int> SrcRule;   //记录源端规则
};

class PhraseProTrie
{
public:
	PhraseProTrie()
	{
		_phrasePro.clear();
		_subPhraseTrie.clear();
		_phraseProMap.clear();
		m_ProcFlag = 0;
	//	CandsNum = 0;
	}

	//返回节点的值
	/*vector<vector<s_PhrasePro> > &value()
	{
		return _phrasePro;
	};*/
	vector<s_PhrasePro> &value()
	{
		return _phrasePro;
	};
	map<string, vector<s_PhrasePro> > & map_value()
	{
		return _phraseProMap;
	};

	map<string, PhraseProTrie> & SubTrie()
	{
		return _subPhraseTrie;
	}
	//插入一个节点
	PhraseProTrie* InsertTrie(const string &Index, PhraseProTrie &newTrie)
	{
		_subPhraseTrie.insert(map<string, PhraseProTrie>::value_type(Index,newTrie) );
		return &_subPhraseTrie[Index];
	};
	
	//查找一个节点;
	PhraseProTrie* findTrie(const string &Index)         
	{
		map<string ,PhraseProTrie>::iterator it= _subPhraseTrie.find(Index);
		if( _subPhraseTrie.end() != it )
			return &( (*it).second );
		else
			return NULL;
	};

	PhraseProTrie* matchIndices(vector<string> &Indices);

//	PhraseProTrie* findTrie(const int *Indices);      //查找一个多层节点,没有则插入
	PhraseProTrie* findTrie(vector<string> & Indices); //查找一个多层节点,没有则插入

	//把_phrasePro中的信息按目标端叶节点放入_phraseProMap中,并完成排序。
	bool RuleProc();
private:

	//存储一个源规则对应的所有目标规则
	map<string, vector<s_PhrasePro> > _phraseProMap;
	vector<s_PhrasePro> _phrasePro;

	//存储节点, 由于tree2tree中需要匹配源端树结构，因此不能用词号作为索引键
	map<string, PhraseProTrie> _subPhraseTrie;
public:
	int m_ProcFlag;                     //二值标志，表示对应的规则集是否已处理过
//	int CandsNum;
};

#endif //!define PHRASEPRO_H

