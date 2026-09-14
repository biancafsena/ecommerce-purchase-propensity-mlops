# E-commerce Purchase Propensity MLOps

Pipeline completo de Machine Learning para estimar a propensão de compra de usuários de um e-commerce com base em seu comportamento de navegação.

## Identificação

- **Instituição:** FIAP
- **Curso:** Pós-Tech Machine Learning Engineering
- **Tech Challenge:** Fase 2
- **Grupo:** 24
- **Autora:** Bianca Firmino Ferreira de Sena
- **Repositório:** `ecommerce-purchase-propensity-mlops`

## Problema de negócio

Uma empresa de e-commerce precisa identificar quais sessões de navegação possuem maior probabilidade de resultar em compra. Essa previsão pode apoiar a priorização de campanhas, ofertas e ações de relacionamento, reduzindo abordagens pouco eficientes.

## Objetivo

Construir uma solução de classificação binária reproduzível, auditável e preparada para produção, com foco em Engenharia de Machine Learning.

O projeto contempla:

- código modular seguindo princípios de Clean Code;
- gerenciamento de dependências com Poetry;
- versionamento de dados e pipeline com DVC;
- rastreamento de experimentos com MLflow;
- gerenciamento de modelos com MLflow Model Registry;
- containerização com Docker;
- testes automatizados e validações de qualidade;
- documentação das decisões técnicas;
- API de inferência preparada para deploy;
- avaliação de probabilidade, calibração e threshold de negócio.

## Dataset

Será utilizado o **Online Shoppers Purchasing Intention Dataset**, disponibilizado pelo UCI Machine Learning Repository.

O alvo `Revenue` indica se a sessão de navegação terminou ou não em uma compra.

## Tecnologias

- Python 3.11
- Scikit-Learn
- Pandas
- MLflow
- DVC
- Poetry
- Docker
- Pytest
- Ruff
- FastAPI

## Status

> Em desenvolvimento — estruturação inicial do projeto.

## Princípios do projeto

- Reprodutibilidade desde a origem.
- Separação clara de responsabilidades.
- Prevenção de vazamento de dados.
- Métricas adequadas ao desbalanceamento.
- Rastreabilidade entre código, dados, experimentos e modelos.
- Decisões técnicas justificadas pelo contexto de negócio.

## Autora

**Bianca Firmino Ferreira de Sena**

Grupo 24 — Pós-Tech Machine Learning Engineering — FIAP.
