#include <xerus.h>
#include <math.h>       /* log2 */

using namespace xerus;
using xerus::misc::operator<<;


class ContractionTree {
	public:
		const size_t d;
		std::vector<std::vector<Tensor>> tree;
		const std::vector<size_t> sample;
		const TTTensor phi;
	private:
		const size_t lvl;


	//function
	public:
		ContractionTree(TTTensor &_phi, std::vector<size_t> _sample) : phi(_phi),d(_phi.order()), sample(_sample),
			lvl(std::ceil(std::log2(_phi.order()))+1){
			XERUS_LOG(info,"d = " << d);
			XERUS_LOG(info,"Sample = " << sample);
			XERUS_LOG(info,"lvl = " << lvl);
			Index r1,r2,r3,r4;

			XERUS_LOG(info,"Initialize empty tree");
			for (size_t i; i < lvl; ++i){
				tree.emplace_back(std::vector<Tensor>());
			}
			XERUS_LOG(info,"Initialize leaves");
			for (size_t i = 0; i < d; ++i){
				auto comp = phi.get_component(i);
				if (std::find(sample.begin(), sample.end(), i) != sample.end())
					comp.fix_mode(1,1);
				else
					comp.fix_mode(1,0);
				tree[0].emplace_back(comp);
			}
			XERUS_LOG(info,"Build tree");
			for (size_t l = 1; l < lvl; ++l){
				size_t s = tree[l-1].size() / 2;
				for (size_t c = 0; c < s; ++s){
					Tensor tmp;
					tmp(r1,r2) = tree[l-1][2*s](r1,r3) * tree[l-1][2*s+1](r3,r2);
					tree[l].emplace_back(std::move(tmp));
				}
				if (2*s < tree[l-1].size())
					tree[l].emplace_back(tree[l].back());

			}
			XERUS_LOG(info,tree[lvl-1][0][0]);
		}


		//ContractionTree(const std::vector<std::vector<Tensor>> tree);
		//value_t getValue();
		//ContractionTree updatedTree(std::vector<size_t> new_sample);


};
