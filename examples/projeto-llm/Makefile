.PHONY: dev run bench test lint docker clean

dev:
	uvicorn turboserve.api.server:app --reload --host 0.0.0.0 --port 8000

run:
	python -m turboserve.api.server

bench:
	python -m turbobench.scripts.benchmark

test:
	python -m pytest tests/ -v

lint:
	python -m py_compile turbolang/parser/lexer.py
	python -m py_compile turbolang/parser/parser.py
	python -m py_compile turboir/ir/nodes.py
	python -m py_compile turboserve/api/server.py

compile:
	python -c "from turbolang.parser.lexer import Lexer; from turbolang.parser.parser import Parser; from turbolang.validator.validator import Validator; from turboir.ir.nodes import IRGenerator; from turboir.planner.planner import ExecutionPlanner; src=open('$(FILE)').read(); tokens=Lexer(src).tokenize(); ast=Parser(tokens).parse(); Validator().validate(ast); ir=IRGenerator().generate(ast); plan=ExecutionPlanner().plan(ir); print(plan)"

docker:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf turbobench/reports/*.json turbobench/reports/*.csv
