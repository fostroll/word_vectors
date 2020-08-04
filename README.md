# word_vectors

Training and evaluation pipelines for *Glove* and *Word2vec*.

## ����������

����������� Python ������ 3 � ����� `gensim`. ��� ���������� ��������� �����
������ `corpuscula` � `toxine`.

## �������������

```sh
python parse_wiki.py
```
���������� ��������� ��� ������������ �� ���� ������� ���������. �� ������ -
���� � ����������������� ������������� (������ ��������� ��������). ����
�������� � ����� *FORM* ��� *LEMMA* � � ���� *MISC* ���������� �������
*Entity*, �� �������� ����� �� `'<EntityX>'`, ��� X - �������� ��������.
������ � ������������� ��������� ����� ��������� �� ����. ������ ���� `<Y>`,
��� Y - ��� ����. ������.

�������� ����� �����. ������������� ������� ������� � �������������� ���������
���������� `corpuscula` � `toxine`, � ����� ��������� � ������ � ����������
`corpus` ��� ������������� *CoNLL-U*-�����. ����� ���� ������������� ����
����� ����� ������������ � ���������� ��� ������ �����.

������ �������� `glove`:
```sh
sh run_glove_train.sh
```
���������� �������� *Glove*. ������������ `1000` ����, ���������� ����� ������
`20` (��. ���������� `EPOCHS` � `CHECKPOINT_EVERY` � �������).

������ �������� `w2v`:
```sh
sh run_w2v_train.sh
```
���������� �������� *Word2vec*. ������������ `1000` ����, ���������� �����
������ `20` (��. ���������� `EPOCHS`, `MAX_EPOCHS` � `CHECKPOINT_EVERY` �
�������).

�� ��������� ������������ ����� *CBOW*. ����� �������� *Skip-gram*, ����������
���������� `SG` � `1`.

� *CBOW* �� ��������� ����������� ������� �����������. ���� �����
������������, ���������� ���������� `CBOW_SUM` � `1`.

---

���������� �������� ����� � ������ ����������� ��������� ��� ������
��������������� ����������, ���������� � �������� `wv_evaluate.ipynb`.

## License

***word_vectors*** is released under the Creative Commons License. See the
[LICENSE](https://github.com/fostroll/word_vectors/blob/master/LICENSE) file
for more details.
