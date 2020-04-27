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
		ContractionTree(const TTTensor &_phi, std::vector<size_t> _sample) : phi(_phi),d(_phi.order()), sample(_sample),
			lvl(std::ceil(std::log2(_phi.order()))+1){

			Index r1,r2,r3,r4;


			std::vector<Tensor> list;
			for (size_t i = 0; i < d; ++i){
				auto comp = phi.get_component(i);
				if (std::find(sample.begin(), sample.end(), i) != sample.end())
					comp.fix_mode(1,1);
				else
					comp.fix_mode(1,0);
				list.emplace_back(std::move(comp));
			}
			size_t noo = 0;
			tree.emplace_back(std::move(list));
			for (size_t l = 1; l < lvl; ++l){
				std::vector<Tensor> list_tmp;
				size_t s = tree[l-1].size() / 2;
				for (size_t c = 0; c < s; ++c){
					Tensor tmp;
					tmp(r1,r2) = tree[l-1][2*c](r1,r3) * tree[l-1][2*c+1](r3,r2);
					noo+= tree[l-1][2*c].dimensions[0]*tree[l-1][2*c].dimensions[1]*tree[l-1][2*c+1].dimensions[1];
					list_tmp.emplace_back(std::move(tmp));
				}
				if (2*s < tree[l-1].size())
					list_tmp.emplace_back(tree[l-1].back());
				tree.emplace_back(std::move(list_tmp));
			}
			//XERUS_LOG(info,"Number of operations " << noo);
		}

		ContractionTree(const TTTensor & _phi, std::vector<size_t> _sample, std::vector<std::vector<Tensor>> _tree) :
			phi(_phi),d(_phi.order()), sample(_sample),lvl(std::ceil(std::log2(_phi.order()))+1), tree(_tree){}

		ContractionTree( const ContractionTree&  _other ) = default;

		value_t getValue(){
			return tree[lvl-1][0][0];
		}

		ContractionTree updatedTree(std::vector<size_t> new_sample){
			Index r1,r2,r3,r4;
			std::vector<std::vector<bool>> update_tree;
			std::vector<std::vector<Tensor>> new_tree = tree;
			std::vector<bool> list;
			for (size_t i = 0; i < d; ++i){
				bool in_old = std::find(sample.begin(), sample.end(), i) != sample.end();
				bool in_new = std::find(new_sample.begin(), new_sample.end(), i) != new_sample.end();
				if (in_old and not in_new){
					auto comp = phi.get_component(i);
					comp.fix_mode(1,0);
					new_tree[0][i] = std::move(comp);
					list.emplace_back(true);
				}
				else if (in_new and not in_old){
					auto comp = phi.get_component(i);
					comp.fix_mode(1,1);
					new_tree[0][i] = std::move(comp);
					list.emplace_back(true);
				}
				else
					list.emplace_back(false);
			}
			update_tree.emplace_back(list);

			size_t noo = 0;
			for (size_t l = 1; l < lvl; ++l){
				std::vector<bool> list_tmp;
				size_t s = tree[l-1].size() / 2;
				for (size_t c = 0; c < s; ++c){
					Tensor tmp;
					if (update_tree[l-1][2*c] or update_tree[l-1][2*c+1]){
						tmp(r1,r2) = new_tree[l-1][2*c](r1,r3) * new_tree[l-1][2*c+1](r3,r2);
						noo+= tree[l-1][2*c].dimensions[0]*tree[l-1][2*c].dimensions[1]*tree[l-1][2*c+1].dimensions[1];
						new_tree[l][c] = std::move(tmp);
						list_tmp.emplace_back(true);
					}
					else
						list_tmp.emplace_back(false);
				}
				if (2*s < new_tree[l-1].size() and update_tree[l-1].back()){
					list_tmp.emplace_back(true);
					new_tree[l].back() = new_tree[l-1].back();

				}
				update_tree.emplace_back(std::move(list_tmp));
			}
			//XERUS_LOG(info,"Number of operations " << noo);

			return ContractionTree(phi,new_sample,new_tree);
		}


};
