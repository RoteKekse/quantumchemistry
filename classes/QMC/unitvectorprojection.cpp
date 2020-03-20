#include <xerus.h>
#include <stdlib.h>

using namespace xerus;
using xerus::misc::operator<<;

#pragma once

class unitVectorProjection{
	public:
		std::pair<TTTensor, TTTensor> xbase;
		size_t d;
	private:
		Tensor leftxStack,rightxStack;
	public:
		unitVectorProjection(TTTensor _psi) : d(_psi.order()){
			_psi.move_core(0,true);
			xbase.first = _psi;
			_psi.move_core(d-1,true);
			xbase.second = _psi;
		}

		void update(TTTensor psi){
			psi.move_core(0,true);
			xbase.first = psi;
			psi.move_core(d-1,true);
			xbase.second = psi;
		}




		Tensor localProduct(std::vector<size_t> sample, size_t position, bool proj = true) {
	  	Tensor comp,tmp,unit,xi;
			Index i1,i2,i3,A1,A2,A3,A4,k1,k2,j1,j2,j3,j4,j5,j6;

			leftxStack = Tensor::ones({1});
			rightxStack = Tensor::ones({1});

			for (size_t pos=d-1; pos>position;pos--){
				tmp = xbase.first.get_component(pos);
				if (std::find(sample.begin(), sample.end(), pos) != sample.end())
					tmp.fix_mode(1,1);
				else
					tmp.fix_mode(1,0);
				rightxStack(i1) = tmp(i1,j1)*rightxStack(j1);
			}
			for (size_t pos=0; pos<position;pos++){
				tmp = xbase.second.get_component(pos);
				if (std::find(sample.begin(), sample.end(), pos) != sample.end())
					tmp.fix_mode(1,1);
				else
					tmp.fix_mode(1,0);
				leftxStack(i1) = tmp(j1,i1)*leftxStack(j1);
			}
			// projection
			if (std::find(sample.begin(), sample.end(), position) != sample.end())
				unit = Tensor::dirac({2},1);
			else
				unit = Tensor::dirac({2},0);

			comp(i1,i2,i3) = leftxStack(i1) * unit(i2)  * rightxStack(i3);

			if (position < d - 1 and proj){
				xi=xbase.second.get_component(position);
				tmp(i1,i2,i3)= xi(i1,i2,k1)*xi(j1,j2,k1)*comp(j1,j2,i3);
				comp -= tmp;
			}
			return comp;
		}

		TTTensor makeUnitVector(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
			return unit;
		}
};
