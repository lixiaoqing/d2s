#include "myutils.h"

vector<string> Split(const string &s)
{
	vector <string> vs;
	stringstream ss(s);
	string e;
	while(ss >> e)
		vs.push_back(e);
	return vs;
}

vector<string> Split(const string &s, const string &sep)
{
	vector <string> vs;
	int cur = 0,next;
	next = s.find(sep);
	while(next != string::npos)
	{
		if(s.substr(cur,next-cur) !="")
			vs.push_back(s.substr(cur,next-cur));
		cur = next+sep.size();
		next = s.find(sep,cur);
	}
	vs.push_back(s.substr(cur));
	return vs;
}

void TrimLine(string &line)
{
	line.erase(0,line.find_first_not_of(" \t\r\n"));
	line.erase(line.find_last_not_of(" \t\r\n")+1);
}

/**************************************************************************************
 1. 函数功能: 将两个span合并为一个span
 2. 入口参数: 被合并的两个span
 3. 出口参数: 合并后的span
 4. 算法简介: 见注释
************************************************************************************* */
Span merge_span(Span span1,Span span2)
{
	if (span2.first == -1)
		return span1;
	if (span1.first == -1)
		return span2;
	Span span;
	span.first = min(span1.first,span2.first);												// 合并后span的左边界
	span.second = max(span1.first+span1.second,span2.first+span2.second) - span.first;		// 合并后span的长度，即右边界减去左边界
	return span;
}

