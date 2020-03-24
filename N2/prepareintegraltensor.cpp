#include <xerus.h>


#include "../classes/loading_tensors.cpp"


int main(){
	size_t nob = 60;
	Tensor T,V, Tn({nob,nob}), Vn({nob,nob,nob,nob});
	read_from_disc("data/T_N2_60_single.tensor", T);
	read_from_disc("data/V_N2_60_single.tensor", V);

	for (size_t i = 0; i < nob; ++i)
		for (size_t j = 0; j <= i; ++j)
			Tn[{i,j}] = T[{i,j}];

	for (size_t i = 0; i < nob; i++)
		for (size_t j = 0; j <= i; j++)
			for (size_t k = 0; k<= i; k++)
				for (size_t l = 0; l <= (i==k ? j : k); l++)
					Vn[{i,j,k,l}] = V[{i,k,j,l}];

	write_to_disc("data/T_N2_60_single_small.tensor", Tn);
	write_to_disc("data/V_N2_60_single_small.tensor", Vn);
	return 0;
}
