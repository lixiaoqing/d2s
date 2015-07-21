#include "stdafx.h"

typedef pair<int,int> Span;			//由起始位置和span长度表示（span长度为实际长度减1）
void TrimLine(string &line);
vector<string> Split(const string &s);
vector<string> Split(const string &s, const string &sep);
Span merge_span(Span span1,Span span2);
