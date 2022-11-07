## ���������� �� ��������� ���������� �� ASTRA Linux

1. ������� Miniconda3 (������: https://docs.conda.io/en/latest/miniconda.html)

2. ������� � �������, ���� ��� ������ shell ������

3. ��������� ������� �� ������� �������: bash Miniconda3-latest-Linux-x86_64.sh

4. ������������� ��������

5. ��������� � ������������� ������: conda list

6. ��������� � ������������� ������ python 3.9: python �version

7. �������� �����������: sudo apt-get update

8. ���������� ������� �������� ������ Git: sudo apt-get install git

9. ����������� ���������: git clone https://github.com/InnopolisUni/innofw

10. ���������� poetry: conda install -c conda-forge poetry

11. ������������� ��������

12. ������� � �����, ���� ���������� ���������: cd innofw

13. ��������� ������� �� ��������� ����������� ����� (������������ ����������� ����� ��� ������ � poetry ���������� ������ ������): poetry shell

14. ���������� ������, ��������� � ����� poetry.lock ��������: poetry install 15. ���������� CUDA � ������������ ����� (https://developer.nvidia.com/cuda-downloads

16. ���� ���������� �������� ������ ��� ������ �������, ���������� �� ���������, �� ���������� ������ ��������� � ���� pyproject.toml � �����, ���� ��� ����������� ������. � ����� ������� ���� poetry.lock � ��� �� �����, ���� �� ��� ������������