NVCC = nvcc

reduction_parallel: headers.h kernel.cu   
	$(NVCC) -I ./ host.cu kernel.cu -o reduction_parallel
run:
	./reduction_parallel
clean:
	rm -rf reduction_parallel