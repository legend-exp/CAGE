# edit_endpoints.py

import sys, os, logging
import psycopg2 as db


db_url = None
db_conn = None


def _initialize(params):
    global db_url
    
    db_url = params.get('db_url', None)
    if db_url is None:
        return
    if db_url[0:13] != 'postgresql://':
        db_url = 'postgresql://' + db_url
        
        

def _finalize():
    global db_conn
    if db_conn is not None:
        db_conn.close()


        
def _process_command(doc):
    global db_conn
    
    if not doc.get('manage_endpoint', False):
        return None
    
    if doc.get('confirm', False)==False or doc.get('confirm2', False)==False or doc.get('confirm3', True)==True:
        return { "status": "error", "message": 'You have to be responsible for what you are doing.' }
    
    endpoint_id = doc.get('endpoint_id', '')
    endpoint_name = doc.get('endpoint_name', '')
    value_type = doc.get('type', 'numeric')
    if not (endpoint_id == '' or endpoint_id.isdigit()):
        return { "status": "error", "message": 'bad Endpoint ID' }
    if endpoint_name == '' or not endpoint_name.replace('_', '').isalnum():
        return { "status": "error", "message": 'bad Endpoint Name' }
    if value_type not in ['numeric', 'json']:
        return { "status": "error", "message": 'bad Endpoint type' }

    if db_conn is False:
        return False
    if db_conn is None:
        try:
            db_conn = db.connect(db_url)
        except Exception as e:
            db_conn = False
            logging.error(e)
            return False

    cursor = db_conn.cursor()
    
    if endpoint_id != '':
        # chaning the endpoint name: check the endpoind ID exists
        sql = "SELECT endpoint_name from endpoint_id_map where endpoint_id='%s'" % endpoint_id
        try:
            cursor.execute(sql)
        except Exception as e:
            logging.error('SQL Error: %s' % str(e))
            return { "status": "error", "message": str(e) }
        if cursor.rowcount != 1:
            return { "status": "error", "message": 'unable to find Endpont ID %s' % endpoint_id }

    # check the name does not already exist
    sql = "SELECT endpoint_id from endpoint_id_map where endpoint_name='%s'" % endpoint_name
    try:
        cursor.execute(sql)
    except Exception as e:
        logging.error('SQL Error: %s' % str(e))
        return { "status": "error", "message": str(e) }
    if cursor.rowcount != 0:
        return { "status": "error", "message": 'Endpont Name already exists: %s' % endpoint_name }
    
    if endpoint_id == '':
        sql = "INSERT INTO endpoint_id_map(endpoint_name, type) VALUES ('%s', '%s')" % (endpoint_name, value_type)
    else:
        sql = "UPDATE endpoint_id_map SET endpoint_name='%s', type='%s' WHERE endpoint_id='%s'" % (endpoint_name, value_type, endpoint_id)

    logging.info(sql)
    try:
        cursor.execute(sql)
        db_conn.commit()
    except Exception as e:
        logging.error('SQL Error: %s' % str(e))
        return { "status": "error", "message": str(e) }

    return True
    
