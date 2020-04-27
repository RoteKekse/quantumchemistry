#include <xerus.h>
#include <math.h>       /* log2 */

using namespace xerus;
using xerus::misc::operator<<;


class ContractionTree {
	public:
		const size_t d;
		const std::vector<std::vector<Tensor>> tree;
		const std::vector<size_t> sample;
	private:
		const size_t lvl;


	//function
	public:
		ContractionTree(TTTensor _phi, std::vector<size_t> _sample) : d(_phi.order()), sample(_sample),
			lvl(std::ceil(std::log2(_phi.order()))+1){
			XERUS_LOG(info,"d = " << d);
			XERUS_LOG(info,"Sample = " << sample);
			XERUS_LOG(info,"lvl = " << lvl);

		}


		//ContractionTree(const std::vector<std::vector<Tensor>> tree);
		//value_t getValue();
		//ContractionTree updatedTree(std::vector<size_t> new_sample);


};
