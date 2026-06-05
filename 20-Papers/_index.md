---
tags: [index, papers]
---

# 📚 论文索引

按主题浏览：[[20-Papers/WM]] · [[20-Papers/WAM]]

## 📖 在读

```dataview
TABLE WITHOUT ID
  file.link AS "论文",
  authors AS "作者",
  year AS "年份",
  rating AS "评分"
FROM "20-Papers"
WHERE status = "reading"
SORT file.mtime DESC
```

## ✅ 已读

```dataview
TABLE WITHOUT ID
  file.link AS "论文",
  authors AS "作者",
  year AS "年份",
  rating AS "评分",
  date-read AS "读完日期"
FROM "20-Papers"
WHERE status = "read"
SORT date-read DESC
```

## 📋 待读

```dataview
TABLE WITHOUT ID
  file.link AS "论文",
  authors AS "作者",
  year AS "年份"
FROM "20-Papers"
WHERE status = "to-read"
SORT file.ctime DESC
```

## 📊 本月统计

```dataview
LIST length(rows) AS "篇数"
FROM "20-Papers"
WHERE status = "read" AND date-read >= date(today) - dur(30 days)
GROUP BY "已读"
```
