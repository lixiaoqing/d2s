/*
PhraseProTrie.cpp
*/


#include "PhraseProTrie.h"

string Int2Str(int a)
{
	ostringstream s;
	s << a;
	return s.str(); //将int转化为string
}

bool 
s_PhrasePro::operator <(const s_PhrasePro &right)const
{
//	for(int i=0; i<NodeIndexVec.size(); i++)
//		strRule = I2S(viEnPhrase[i])+I2S(NodeIndexVec[i]);
//	for(int i=0; i<right.NodeIndexVec.size(); i++)
//		right.strRule = I2S(right.viEnPhrase[i])+I2S(right.NodeIndexVec[i]);

//	if(strRule == right.strRule)
	return dPro < right.dPro;
//	else
//	return strRule < right.strRule;
}

s_PhrasePro&
s_PhrasePro::operator =(const s_PhrasePro &right)
{
	ulEnNum = right.ulEnNum;					
	viEnPhrase = right.viEnPhrase;
	NodeIndexVec = right.NodeIndexVec;
	strRule = right.strRule;
	SrcRule = right.SrcRule;
	//viLeftofLeftEn = right.viLeftofLeftEn;
	//viRightofLeftEn = right.viRightofLeftEn;
	//viRightofRightEn = right.viRightofRightEn;
	//invertedFlag = right.invertedFlag;
	composedFlag = right.composedFlag;
	LexFlag = right.LexFlag;
	rootIndex = right.rootIndex;
	dPro = right.dPro;						
	eachTransPro = right.eachTransPro;
	ngramPro = right.ngramPro;
	return *this;
}

////////////////////////////////////////////////////////////////////////
// 1.主要功能：查找多层节点（按照中文短语查找，每一层存储一个中文词）
// 2.入口参数：中文短语的词号
// 3.出口参数：节点的指针
// 4.生成日期：2006.04.10
// 5.算法简介：
//
/*
PhraseProTrie* 
PhraseProTrie::findTrie(const int *Indices)
{
	int m_len= Vocab::length(Indices);
        //cout<<m_len<<endl;
	PhraseProTrie* subTrie=this;
	for(int i=0; i<m_len; i++)
	{
		PhraseProTrie* m_subTemp=subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
		{
			PhraseProTrie m_triTemp;
			subTrie=subTrie->InsertTrie(Indices[i],m_triTemp);
		}
	}
	return subTrie;
}
*/

PhraseProTrie * PhraseProTrie::findTrie(vector<string> & Indices)
{
	int m_len= Indices.size();
	//cout<<m_len<<endl;
	PhraseProTrie* subTrie=this;
	for(int i=0; i<m_len; i++)
	{
		PhraseProTrie* m_subTemp=subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
		{
			PhraseProTrie m_triTemp;
			subTrie=subTrie->InsertTrie(Indices[i],m_triTemp);
		}
	}
	return subTrie;
}

//该函数只负责查找，不负责插入操作
PhraseProTrie* PhraseProTrie::matchIndices(vector<string> &Indices)
{
	int m_len = Indices.size();
	PhraseProTrie* subTrie = this;
	for( int i = 0; i < m_len; i++ )
	{
		PhraseProTrie* m_subTemp = subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
			return NULL;
	}
	return subTrie;
}

bool PhraseProTrie::RuleProc()
{
	if(!m_ProcFlag)
	{
		vector<s_PhrasePro>::iterator iter = _phrasePro.begin();
		for(; iter != _phrasePro.end(); ++iter)
		{
/*			for(int j=0; j<iter->NodeIndexVec.size(); j++)
				cout<<(iter->viEnPhrase[j])<<" ";
			cout<<this<<endl;
*/
			//生成表征规则的字符串，
			//字符串可以是只有叶节点Tag，也可以再包含与源端的索引，两者得到不同map
			//此处都使用
			if(iter->strRule == "")
			{
				string strRule = "";
				for(int i=0; i<(iter->NodeIndexVec).size(); i++)
				{
					if(iter->NodeIndexVec[i] == -1)
						continue;
					strRule = strRule + Int2Str((iter->viEnPhrase)[i]) +"-"+Int2Str((iter->NodeIndexVec)[i])+"@";
				}
				iter->strRule = strRule;
			}

			//加入到Map中
			map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.find(iter->strRule);
			if(m_iter == _phraseProMap.end())
			{
				vector<s_PhrasePro> c_PhraseProVec;
				c_PhraseProVec.push_back(*iter);
				_phraseProMap.insert(map<string, vector<s_PhrasePro> >::value_type(iter->strRule, c_PhraseProVec));
			}
			else
				m_iter->second.push_back(*iter);
		}

		//对Map中的规则按打分进行排序
		map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.begin();
		for(; m_iter!=_phraseProMap.end(); ++m_iter)
		{
			sort(m_iter->second.rbegin(), m_iter->second.rend());
		}

		m_ProcFlag = 1;
	}
}

/*
bool PhraseProTrie::UnaryProc()
{
	if(!m_UnaryFlag)
	{
		vector<s_PhrasePro>::iterator iter = _phrasePro.begin();
		for(; iter != _phrasePro.end(); ++iter)
		{
			int VarNum = 0;
			int VarIndex = -1;
			for(int j=0; j<iter->NodeIndexVec.size(); j++)
			{
				if(NodeIndexVec[j] == -1)
				   continue;	
				
				VarNum++;
				if(VarNum > 1)
					break;
				VarIndex = _phrasePro.viEnPhrase[NodeIndexVec[i]];
			}
			if(VarNum != 1)
				continue;

			map<int, vector<s_PhrasePro*> >::iterator m_iter = _UnaryProMap.find(VarIndex);
			if(m_iter == _UnaryProMap.end())
			{
				vector<s_PhrasePro*> c_PhraseProVec;
				c_PhraseProVec.push_back(&(*iter));
				_UnaryProMap.insert(map<int, vector<s_PhrasePro*> >::value_type(int, c_PhraseProVec));
			}
			else
				m_iter->second.push_back(&(*iter));
		}

	//	map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.begin();
	//	for(; m_iter!=_phraseProMap.end(); ++m_iter)
	//	{
	//		sort(m_iter->second.rbegin(), m_iter->second.rend());
	//	}

	//	_phrasePro.clear();
		m_UnaryFlag = 1;
	}
}*/
