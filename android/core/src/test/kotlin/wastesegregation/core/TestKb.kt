package wastesegregation.core

import java.io.File
import java.sql.DriverManager

/** Shared test setup: the repo root, the compiled kb.sqlite (read with JDBC) and fixtures. */
object TestKb {
    val repoRoot: File = File(System.getProperty("repo.root") ?: error("repo.root not set (run via Gradle)"))

    @Suppress("UNCHECKED_CAST")
    private fun obj(json: String) = Json.parse(json) as Map<String, Any?>

    val kb: KnowledgeBase by lazy {
        val db = repoRoot.resolve("kb/build/kb.sqlite")
        DriverManager.getConnection("jdbc:sqlite:${db.absolutePath}").use { con ->
            fun rows(sql: String): List<List<String>> = con.createStatement().use { st ->
                st.executeQuery(sql).use { rs ->
                    val n = rs.metaData.columnCount
                    buildList { while (rs.next()) add((1..n).map { rs.getString(it) }) }
                }
            }
            val meta = rows("SELECT key, value FROM meta").associate { it[0] to it[1] }
            KnowledgeBase(
                KbData(
                    items = rows("SELECT id, json FROM items").associate { it[0] to obj(it[1]) },
                    cities = rows("SELECT id, json FROM cities").associate { it[0] to obj(it[1]) },
                    aliases = rows("SELECT alias, lang, item_id FROM aliases").map { AliasRow(it[0], it[1], it[2]) },
                    detectorClasses = rows("SELECT class, item_id FROM detector_classes").associate { it[0] to it[1] },
                    version = meta.getValue("kb_version"),
                    strings = obj(meta.getValue("strings")),
                ),
            )
        }
    }

    fun readJson(path: String): Any? = Json.parse(repoRoot.resolve(path).readText(Charsets.UTF_8))
}
