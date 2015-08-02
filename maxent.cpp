#include "maxent.h"

void MaxentModel::load(const string &modelfile)
{
	gzFile f = gzopen(modelfile.c_str(), "rb");
	if (f == NULL)
	{
		cerr<<"cannot open model file!\n";
		return;
	}

	char buf[100];
	gzread(f, (void*)buf, 16);
	buf[16] = '\0';
	string s = buf;

	if (s.find("txt") != s.npos) 
	{
		load_txt(modelfile);
	}
       	else if (s.find("bin") != s.npos) 
	{
		load_bin(modelfile);
	} 
	else
	{
		cerr<<"unknown model type!\n";
		return;
	}
	gzclose(f);
	cout<<"load reorder model file "<<modelfile<<" over\n";
}

void MaxentModel::load_bin(const string &modelfile)
{
	gzFile f;
	f = gzopen(modelfile.c_str(), "rb");
	if (f == NULL)
	{
		cerr<<"cannot open model file!\n";
		return;
	}
	//skip header
	gzseek(f, 16, 0);

	char buf[4096]; // TODO: handle unsafe buffer
	// read features
	gzread(f, (void*)&feature_num, sizeof(feature_num));
	size_t len;
	for (size_t i = 0; i < feature_num; ++i) 
	{
		gzread(f, (void*)&len, sizeof(len));
		gzread(f, (void*)&buf, len);
		feature2id.insert(make_pair(string(buf,len),i));
	}

	// read tags
	gzread(f, (void*)&tag_num, sizeof(tag_num));
	for (size_t i = 0; i < tag_num; ++i) 
	{
		gzread(f, (void*)&len, sizeof(len));
		gzread(f, (void*)&buf, len);
		tag2id.insert(make_pair(string(buf,len),i));
	}

	// read paramaters
	size_t tagid;
	featureAddrVec.resize(feature_num+1,0);
	for (size_t i = 0; i < feature_num; ++i) 
	{
		gzread(f, (void*)&len, sizeof(len));
		featureAddrVec.at(i+1) = featureAddrVec.at(i)+len;
		for (size_t j = 0; j < len; ++j) 
		{
			gzread(f, (void*)&tagid, sizeof(tagid));
			lambda2tagVec.push_back(tagid);
		}
	}

	// load theta
	gzread(f, (void*)&lambda_num, sizeof(lambda_num));

	double lambda;
	for (size_t i = 0; i < lambda_num; ++i) 
	{
		gzread(f, (void*)&lambda, sizeof(lambda));
		lambdaVec.push_back(lambda);
	}
	gzclose(f);
}

void MaxentModel::load_txt(const string &modelfile)
{
	ifstream fin;
	fin.open(modelfile.c_str());
	if(!fin.is_open())
	{
		cerr<<"open MaxEnt model error!"<<endl;
		return;
	}
	string s;
	getline(fin,s);
	getline(fin,s);
	feature_num = stoi(s);
	for (int i = 0;i<feature_num;i++)
	{
		getline(fin,s);
		TrimLine(s);
		feature2id.insert(pair<string,int>(s,i));
		featureVec.push_back(s);
	}
	getline(fin,s);
	tag_num = stoi(s);
	for (int i = 0; i<tag_num; i++)
	{
		getline(fin,s);
		TrimLine(s);
		tag2id.insert(make_pair(s,i));
	}
	featureAddrVec.resize(feature_num+1,0);
	for (int i = 0; i<feature_num; i++)
	{
		getline(fin,s);
		TrimLine(s);
		vector <string> vs = Split(s);
		featureAddrVec.at(i+1) = featureAddrVec.at(i)+stoi(vs.at(0));
		for(size_t j = 1;j<vs.size();j++)
			lambda2tagVec.push_back(stoi(vs.at(j)));
	}
	getline(fin,s);
	lambda_num = stoi(s);
	for (int i = 0; i<lambda_num;i++)
	{
		getline(fin,s);
		TrimLine(s);
		double lambda = stod(s);
		lambdaVec.push_back(lambda);
	}
}

int MaxentModel::get_tagid(string tag)
{
	auto it = tag2id.find(tag);
	if (it != tag2id.end())
	{
		return it->second;
	}
	return -1;
}

double MaxentModel::eval(vector<string> &context, string & tag)
{
	int tagid = get_tagid(tag);
	if (tagid == -1)
		return -99;
	vector<double> probs;
	eval_all(probs,context);
	return probs.at(tagid);
}

void MaxentModel::eval_all(vector<double> &me_scores, vector<string> &context)
{
	me_scores.resize(tag_num, 0.0);
	
	for (size_t i = 0; i < context.size(); i ++)
	{
		auto iter = feature2id.find(context.at(i));
		if (iter != feature2id.end())
		{
			int featureid = iter->second;
			int num1 = featureAddrVec.at(featureid);
			int num2 = featureAddrVec.at(featureid+1);
			for (int j = num1; j < num2; j ++)
			{
				me_scores.at(lambda2tagVec.at(j)) += lambdaVec.at(j);
			}
		}
		else
		{
			continue;
		}
	}
	
	double sum = 0.0;
	for (size_t j = 0; j < me_scores.size(); j ++)
	{
		me_scores.at(j) = exp(me_scores.at(j));
		sum += me_scores.at(j);
	}
	
	for (size_t k = 0; k < me_scores.size(); k ++)
	{
		me_scores.at(k) /= sum; 
		me_scores.at(k) = log10(me_scores.at(k)); 
	}
}

