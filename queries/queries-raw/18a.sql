SELECT MIN(mi.info) AS movie_budget,
       MIN(t.title) AS movie_title
FROM cast_info AS ci,
     info_type AS it1,
     info_type AS it2,
     movie_info AS mi,
     name AS n,
     title AS t
WHERE ci.note IN ('(producer)',
                  '(executive producer)')
  AND it1.info = 'budget'
  AND it2.info = 'votes'
  AND n.gender = 'm'
  AND n.name LIKE '%Tim%'
  AND t.id = mi.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mi.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi.info_type_id;

