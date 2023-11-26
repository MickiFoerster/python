# Note: the module name is psycopg, not psycopg3
import psycopg

try:
# Connect to an existing database
    with psycopg.connect("dbname=postgres user=postgres") as conn:

        # Open a cursor to perform database operations
        with conn.cursor() as cur:

            cur.execute("""
                SELECT * FROM mx_floodlight_registrations 
                """)

            row = cur.fetchone()
            while row is not None:
                print(row)
                row = cur.fetchone()

            cur.close()
            conn.close()
except (Exception, psycop.DatabaseError) as error:
    print(error)
