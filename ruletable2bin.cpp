#include "myutils.h"
const int LEN = 4096;

void ruletable2bin(string rule_filename)
{
	unordered_map <string,int> src_vocab;
	unordered_map <string,int> tgt_vocab;
	vector<string> src_vocab_vec = {"[x]"};
	vector<string> tgt_vocab_vec = {"[x]"};
	int src_wid = 1;
	int tgt_wid = 1;
	src_vocab.insert(make_pair("[x]",0));
	tgt_vocab.insert(make_pair("[x]",0));
	gzFile gzfp = gzopen(rule_filename.c_str(),"r");
	if (!gzfp)
	{
		cout<<"fail to open "<<rule_filename<<endl;
		return;
	}
	ofstream fout;
	fout.open("prob.bin",ios::binary);
	if (!fout.is_open())
	{
		cout<<"fail open model file to write!\n";
		return;
	}
	char buf[LEN];
	while( gzgets(gzfp,buf,LEN) != Z_NULL)
	{
		string line(buf);
		vector <string> elements = Split(line,"|||");

		for (auto &e : elements)
		{
			TrimLine(e);
		}
		vector <string> src_words = Split(elements[0]);
		vector <int> src_wids;
		for (const auto &src_word : src_words)
		{
			auto it = src_vocab.find(src_word);
			if (it != src_vocab.end())
			{
				src_wids.push_back(it->second);
			}
			else
			{
				src_wids.push_back(src_wid);
				src_vocab.insert(make_pair(src_word,src_wid));
				src_vocab_vec.push_back(src_word);
				src_wid++;
			}
		}

		vector <string> tgt_words = Split(elements[1]);
		vector <int> tgt_wids;
		for (const auto &tgt_word : tgt_words)
		{
			auto it = tgt_vocab.find(tgt_word);
			if (it != tgt_vocab.end())
			{
				tgt_wids.push_back(it->second);
			}
			else
			{
				tgt_wids.push_back(tgt_wid);
				tgt_vocab.insert(make_pair(tgt_word,tgt_wid));
				tgt_vocab_vec.push_back(tgt_word);
				tgt_wid++;
			}
		}

		vector<string> nt_align = Split(elements[2]);
		vector<int> tgt_nt_idx_to_src_nt_idx;
		for (auto &e : nt_align)
		{
			tgt_nt_idx_to_src_nt_idx.push_back(stoi(e));
		}

		vector <string> prob_str_vec = Split(elements[3]);
		vector <double> prob_vec;
		for (const auto &prob_str : prob_str_vec)
		{
			double prob = stod(prob_str);
			double log_prob = 0.0;
			if( abs(prob) <= numeric_limits<double>::epsilon() )
			{
				log_prob = LogP_PseudoZero;
			}
			else
			{
				log_prob = log10(prob);
			}
			prob_vec.push_back(log_prob);
		}

		int rule_src_len = src_wids.size();
		int rule_tgt_len = tgt_wids.size();
		fout.write((char*)&rule_src_len,sizeof(int));
		fout.write((char*)&src_wids[0],sizeof(int)*rule_src_len);
		fout.write((char*)&rule_tgt_len,sizeof(int));
		fout.write((char*)&tgt_wids[0],sizeof(int)*rule_tgt_len);
		fout.write((char*)&tgt_nt_idx_to_src_nt_idx[0],sizeof(int)*tgt_nt_idx_to_src_nt_idx.size());
		fout.write((char*)&prob_vec[0],sizeof(double)*prob_vec.size());
	}
	gzclose(gzfp);
	fout.close();

	ofstream f_ch_vocab("vocab.ch");
	if (!f_ch_vocab.is_open())
	{
		cout<<"fail open ch vocab file to write!\n";
		return;
	}
	for(size_t i=0;i<src_vocab_vec.size();i++)
	{
		f_ch_vocab<<src_vocab_vec.at(i)+" "+to_string(i)+"\n";
	}
	f_ch_vocab.close();

	ofstream f_en_vocab("vocab.en");
	if (!f_en_vocab.is_open())
	{
		cout<<"fail open en vocab file to write!\n";
		return;
	}
	for(size_t i=0;i<tgt_vocab_vec.size();i++)
	{
		f_en_vocab<<tgt_vocab_vec.at(i)+" "+to_string(i)+"\n";
	}
	f_en_vocab.close();
}

int main(int argc,char* argv[])
{
    if(argc == 1)
    {
		cout<<"usage: ./ruletable2bin ruletable\n";
		return 0;
    }
    string rule_filename(argv[1]);
    ruletable2bin(rule_filename);
	return 0;
}
