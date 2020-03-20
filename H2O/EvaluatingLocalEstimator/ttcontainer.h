#include <xerus.h>
#include <memory>
#include <algorithm>
#include <functional>



using namespace xerus;
using xerus::misc::operator<<;

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    XERUS_REQUIRE(a.size() == b.size(),"size mismatch " << a.size() << " != " << b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}


class TTContainer { //NOTE: rounding the container will destroy the container
	public:
		std::vector<size_t> capacity;
		std::vector<size_t> capacity_used;
	private:
		TTTensor container;
	public:
		TTContainer(size_t _capacity, size_t _dim, size_t order) : TTContainer(std::vector<size_t>(order,_dim), std::vector<size_t>(order - 1,_capacity)){}

		TTContainer(std::vector<size_t> _dimensions, std::vector<size_t> _capacity ) : container(_dimensions), capacity(_capacity), capacity_used(std::vector<size_t>(_dimensions.size()-1,0)) {
			XERUS_REQUIRE(_dimensions.size() == _capacity.size() +1, "_dimension and _capacity do not have the correct size");
			const size_t order = _dimensions.size();
			container.set_component(0,Tensor({1,_dimensions[0],_capacity[0]}));
			for (size_t i = 1; i < order-1; ++i){
				Tensor tmp;
				tmp = Tensor({_capacity[i-1],_dimensions[i],_capacity[i]});
				container.set_component(i,tmp);
			}
			container.set_component(order-1,Tensor({_capacity[order-2],_dimensions[order-1],1}));
		}

		TTContainer(const TTTensor _tt, size_t _capacity) : TTContainer(_tt, std::vector<size_t>(_tt.order() - 1,_capacity)) {}
		TTContainer(const TTTensor _tt, std::vector<size_t> _capacity): TTContainer(_tt.dimensions, _capacity + _tt.ranks()){
			capacity_used = _tt.ranks();
			for (size_t i = 0; i < _tt.order(); ++i){
				container.component(i).offset_add(_tt.get_component(i),std::vector<size_t>({0,0,0}));
			}
		}
		TTTensor get_container(){
			XERUS_LOG(info,container.ranks());
			return container;
		}

		bool isFull(std::vector<size_t> _new){
			auto state = capacity_used + _new;
			for (size_t i = 0; i < capacity.size(); ++i){
				if (state[i] > capacity[i]){
					return true;
				}
			}
			return false;
		}
		void resize(size_t _capacity){ //resizes the container, _capacity is added to the size
			resize(std::vector<size_t>(container.order() - 1,_capacity));
		}
		void resize(std::vector<size_t> _capacity){ //resizes the container
			for (size_t i = 0; i < container.order(); ++i){
				XERUS_REQUIRE(capacity[i] <= _capacity[i], "new size is smaller than before");
				Tensor tmp;
				if (i == 0)
					tmp = Tensor({1,container.dimensions[i],_capacity[i]});
				else if (i == container.order() - 1)
					tmp = Tensor({_capacity[i-1],container.dimensions[i],1});
				else
					tmp = Tensor({_capacity[i-1],container.dimensions[i],_capacity[i]});
				tmp.offset_add(container.get_component(i),std::vector<size_t>({0,0,0}));
				container.set_component(i,tmp);
			}
		}
		void addTT(TTTensor _tt2, size_t _dim){
			XERUS_LOG(test,isFull(_tt2.ranks()));
			if (isFull(_tt2.ranks()))
				resize(capacity + capacity + _tt2.ranks());
			XERUS_PA_START;
			for(size_t position = 0; position < _dim; ++position) {
				// Get current components
				const Tensor& otherComponent = _tt2.get_component(position);

				const size_t leftOffset = position == 0 ? 0 : capacity_used[position - 1];
				const size_t rightOffset = position == _dim-1 ? 0 : capacity_used[position];

				container.component(position).offset_add(otherComponent, std::vector<size_t>({leftOffset,0,rightOffset}));
			}
			XERUS_PA_END("ADD/SUB", "TTNetwork ADD/SUB", std::string("Dims:")+misc::to_string(dimensions)+" Ranks: "+misc::to_string(ranks()));
			for (size_t i = 0; i < _dim;++i){
				capacity_used[i] +=  0;
			}
	}

	void partial_round(size_t dim){
		for (size_t n = 0; n < dim-1; ++n) {
			container.transfer_core(n+1, n+2, true);
		}
		auto epsPerSite = EPSILON;
		for(size_t i = 0; i+1 < dim; ++i) {
			container.round_edge(dim-i, dim-i-1, std::vector<size_t>(container.ranks().size(), std::numeric_limits<size_t>::max())[dim-i-2], epsPerSite, 0.0);
		}
	}


};
