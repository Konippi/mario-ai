.PHONY: run-server
run-server:
	@echo "Running server..."
	@cd marioai-2009-yatteiku && \
		ant clean compile && \
		java -cp classes ch.idsia.scenarios.MainRun \
			-server on -vaot on -vlx 1044 -vly 10 \
			-ag ScaredAgent -ld 5 -lt 1

.PHONY: run-client
run-client:
	@echo "Running client..."
	@cd marioai-2009-yatteiku/src && \
		python python/competition/learn.py
