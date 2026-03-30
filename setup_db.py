"""Setup database user and database."""
import psycopg2
import getpass

# Connect to postgres database to create user and database
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user=getpass.getuser()  # Use current system user
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Create user
    try:
        cur.execute("CREATE USER meridian_user WITH PASSWORD 'password';")
        print("✅ Created user: meridian_user")
    except psycopg2.errors.DuplicateObject:
        print("ℹ️  User meridian_user already exists")

    # Create database
    try:
        cur.execute("CREATE DATABASE meridian OWNER meridian_user;")
        print("✅ Created database: meridian")
    except psycopg2.errors.DuplicateDatabase:
        print("ℹ️  Database meridian already exists")

    # Grant privileges
    cur.execute("GRANT ALL PRIVILEGES ON DATABASE meridian TO meridian_user;")
    print("✅ Granted privileges to meridian_user")

    cur.close()
    conn.close()
    print("\n✅ Database setup complete!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Try running with your system user:")
    print("   psql -d postgres -c 'CREATE USER meridian_user WITH PASSWORD \"password\";'")
    print("   psql -d postgres -c 'CREATE DATABASE meridian OWNER meridian_user;'")
