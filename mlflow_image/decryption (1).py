class AccessSecret:
    
    def __init__(self):
        from google.cloud import secretmanager
        self.client = secretmanager.SecretManagerServiceClient()
    
    def unpad(self, s):
        """unpad ciphering string.
        Args:
            s: string
        """
        return s[:-ord(s[len(s)-1:])]
    
    def access_secret(self, project_id, secret_id, version):
        """Access the secret values.
        Args:
            project_id: GCP project ID
            secret_id: GCP Secret Manager secret ID
            version: version of secret key. You can use "latest"
        Returns:
            payload of the variable in Secret Manager.
        """
        name = self.client.secret_version_path(project_id, secret_id, version)
        response = self.client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode('UTF-8')
        return payload
    
    def decrypt(self, key, enc):
        """decript the encoded value with decryption key.
        Args: 
            key: GCP decription key e.g. ELEMENT_DECRYPTION_KEY
            enc: GCP encoding value as project token e.g. ELEMENT_PROJECT_TOKEN
        Returns:
            get the decrypted value as utf-8 form 
        """
        import base64
        import hashlib
        from Crypto.Cipher import AES
        enc = base64.b64decode(enc)
        key = hashlib.sha256(key.encode()).digest()
        iv = enc[:AES.block_size]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return self.unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')
    
    

